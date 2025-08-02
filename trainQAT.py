#!/usr/bin/env python3
"""
HLS4ML Semantic Segmentation Training with QAT Toggle
=====================================================

This script implements the recommended hls4ml approach:
1. Train a baseline model in float32 (trainQAT=False)
2. Load the trained float32 model and apply QAT (trainQAT=True)

Usage:
    python train_qat_toggle.py --trainQAT=False  # Train baseline float32 model
    python train_qat_toggle.py --trainQAT=True   # Apply QAT to trained model

Key Features:
- Baseline float32 training for optimal convergence
- QAT application after float32 training completes
- Same architecture with quantized/non-quantized layer swapping
- Memory-efficient data pipeline
- FPGA-compatible architecture (no MaxPooling, etc.)
"""

import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
import signal
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Multiply, Dropout
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path
import math

# Parse command line arguments
parser = argparse.ArgumentParser(description='HLS4ML Semantic Segmentation with QAT Toggle')
parser.add_argument('--trainQAT', type=str, default='False', 
                   help='Enable QAT training (True/False). Default: False for float32 baseline')
parser.add_argument('--load_model', type=str, default='', 
                   help='Path to float32 model to load for QAT training')
parser.add_argument('--epochs', type=int, default=50, 
                   help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2, 
                   help='Batch size for training')
parser.add_argument('--use_dropout', type=str, default='True',
                   help='Enable dropout during training (True/False). Default: True')

args = parser.parse_args()

# Convert string to boolean
TRAIN_QAT = args.trainQAT.lower() == 'true'
LOAD_MODEL_PATH = args.load_model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
USE_DROPOUT_DURING_TRAINING = args.use_dropout.lower() == 'true'

print(f"Training Configuration:")
print(f"  QAT Training: {TRAIN_QAT}")
print(f"  Load Model: {LOAD_MODEL_PATH if LOAD_MODEL_PATH else 'None (training from scratch)'}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Use Dropout: {USE_DROPOUT_DURING_TRAINING}")

# Define constants
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 34
DATASET_DIR = '/data'
SUBSET_RATIO = 1.0
NUM_WORKERS = 4
PREFETCH_SIZE = 2

DEBUG_MODE = False

# Model architecture configuration
MODEL_CONFIG = {
    'init_filters': INPUT_HEIGHT // 32,  # 16
    'stage1_filters': INPUT_HEIGHT // 16, # 32
    'stage2_filters': INPUT_HEIGHT // 8,  # 64
    'stage3_filters': INPUT_HEIGHT // 4,  # 128
    'stage4_filters': INPUT_HEIGHT // 2,  # 256
}

# Configure TensorFlow
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)

print(f"Eager execution enabled: {tf.executing_eagerly()}")
tf.debugging.set_log_device_placement(DEBUG_MODE)

# Disable layout optimizer to prevent conflicts with QKeras
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})

# QKeras quantization parameters (used only when TRAIN_QAT=True)
quantized_bits_config = {
    'bits': 8,
    'integer': 4,
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

quantized_relu_config = {
    'bits': 8,
    'integer': 4,
    'use_stochastic_rounding': True
}

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Cleanup memory and run garbage collection"""
    gc.collect()
    tf.keras.backend.clear_session()

# Simplified Combo Loss using built-in MeanIoU and constant weights
class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_iou=0.3, alpha_ce=0.7, name='combo_loss'):
        super(ComboLoss, self).__init__(name=name)
        self.alpha_iou = alpha_iou  # Weight for IoU loss
        self.alpha_ce = alpha_ce    # Weight for CrossEntropy loss
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # Use built-in MeanIoU for IoU calculation
        self.miou_metric = tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)

    def call(self, y_true, y_pred):
        # Standard categorical crossentropy
        ce_loss = self.ce_loss(y_true, y_pred)
        
        # Calculate IoU using built-in MeanIoU metric
        # Convert to class indices for MeanIoU calculation
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # Update the metric state
        self.miou_metric.update_state(y_true_classes, y_pred_classes)
        
        # Get current IoU and convert to loss (1 - IoU)
        current_iou = self.miou_metric.result()
        iou_loss = 1.0 - current_iou
        
        # Handle NaN values
        ce_loss = tf.where(tf.math.is_finite(ce_loss), ce_loss, 0.0)
        iou_loss = tf.where(tf.math.is_finite(iou_loss), iou_loss, 0.0)
        
        # Simple weighted combination
        return self.alpha_iou * iou_loss + self.alpha_ce * ce_loss

# Custom callbacks for better training monitoring
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Clean up memory at the end of each epoch"""
        gc.collect()
        # tf.keras.backend.clear_session()  # Removed - causes training slowdown after epoch 2
        # Log memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"\nMemory usage after epoch {epoch}: {memory_usage:.2f} MB")

class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_file = os.path.join(log_dir, 'log.txt')
        os.makedirs(log_dir, exist_ok=True)
        # Ensure the log file exists
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    f.write('')
            except Exception as e:
                print(f"[TrainingLoggerCallback] Could not create log.txt: {e}")
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Use self.epoch_start_time if available, else fallback to previous epoch duration
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_start_time = epoch_end_time  # Prepare for next epoch
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Format metrics for log
        loss = logs.get('loss', None)
        mean_iou = logs.get('mean_iou', None)
        val_loss = logs.get('val_loss', None)
        val_mean_iou = logs.get('val_mean_iou', None)
        # Format time as seconds
        time_str = f"{int(epoch_duration)}s"
        # Compose log line similar to terminal output
        log_line = f"{time_str} 1s/step - "
        if loss is not None:
            log_line += f"loss: {loss:.4f} - "
        if mean_iou is not None:
            log_line += f"mean_iou: {mean_iou:.4f} - "
        if val_loss is not None:
            log_line += f"val_loss: {val_loss:.4f} - "
        if val_mean_iou is not None:
            log_line += f"val_mean_iou: {val_mean_iou:.4f} - "
        log_line += f"lr: {lr:.6e}\n"
        # Log model saving info
        model_prefix = "qat_model" if TRAIN_QAT else "float32_model"
        log_line += f"Epoch {epoch+1}: saving model to ./models/{model_prefix}_checkpoint_epoch_{epoch+1:02d}.keras\n"
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"[TrainingLoggerCallback] Could not write to log.txt: {e}")

class CityscapesDataGenerator:
    """Data generator for Cityscapes dataset"""
    
    def __init__(self, image_dir, mask_dir, batch_size=1, subset_ratio=1.0, 
                 target_height=512, target_width=1024, shuffle=True, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment
        
        # Get all image paths
        self.image_paths = list(self.image_dir.glob('**/*.png'))
        if subset_ratio < 1.0:
            subset_size = int(len(self.image_paths) * subset_ratio)
            self.image_paths = self.image_paths[:subset_size]
        
        print(f"Found {len(self.image_paths)} images")
        
        # Create augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.transform = None
        
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load corresponding mask
            mask_name = image_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            mask_path = self.mask_dir / image_path.parent.name / mask_name
            
            if not mask_path.exists():
                # Try alternative mask naming
                mask_name = image_path.name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                mask_path = self.mask_dir / image_path.parent.name / mask_name
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            # Resize image and mask
            image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
            
            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Convert mask to one-hot encoding
            mask = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            dummy_image = np.zeros((self.target_height, self.target_width, 3), dtype=np.float32)
            dummy_mask = np.zeros((self.target_height, self.target_width, NUM_CLASSES), dtype=np.float32)
            dummy_mask[:, :, 0] = 1  # Set background class
            return dummy_image, dummy_mask
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            batch_images.append(image)
            batch_masks.append(mask)
        
        return np.array(batch_images), np.array(batch_masks)

def conv_block(x, filters, kernel_size=3, strides=1, use_quantization=False, name='conv_block'):
    """
    Convolutional block with option for quantization
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size
        strides: Stride value
        use_quantization: Whether to use QKeras quantized layers
        name: Block name
    """
    if use_quantization:
        # Use QKeras quantized layers
        x = QConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_quantizer=quantized_bits(**quantized_bits_config),
                   bias_quantizer=quantized_bits(**quantized_bits_config),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                   name=f'{name}_qconv')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**quantized_bits_config),
            beta_quantizer=quantized_bits(**quantized_bits_config),
            name=f'{name}_qbn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name=f'{name}_qact')(x)
    else:
        # Use standard float32 layers
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                  name=f'{name}_conv')(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = Activation('relu', name=f'{name}_act')(x)
    
    return x

def attention_block(x, use_quantization=False, name='attention'):
    """
    Channel attention block with option for quantization
    """
    channels = x.shape[-1]
    
    # Global average pooling
    y = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    y = tf.keras.layers.Reshape((1, 1, channels), name=f'{name}_reshape')(y)
    
    # Attention mechanism
    if use_quantization:
        y = QConv2D(channels // 4, kernel_size=1, padding='same',
                   kernel_quantizer=quantized_bits(**quantized_bits_config),
                   bias_quantizer=quantized_bits(**quantized_bits_config),
                   name=f'{name}_qconv1')(y)
        y = QActivation(quantized_relu(**quantized_relu_config), name=f'{name}_qact1')(y)
        y = QConv2D(channels, kernel_size=1, padding='same',
                   kernel_quantizer=quantized_bits(**quantized_bits_config),
                   bias_quantizer=quantized_bits(**quantized_bits_config),
                   name=f'{name}_qconv2')(y)
        y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    else:
        y = Conv2D(channels // 4, kernel_size=1, padding='same',
                  name=f'{name}_conv1')(y)
        y = Activation('relu', name=f'{name}_act1')(y)
        y = Conv2D(channels, kernel_size=1, padding='same',
                  name=f'{name}_conv2')(y)
        y = Activation('sigmoid', name=f'{name}_sigmoid')(y)
    
    return Multiply(name=f'{name}_scale')([x, y])

def create_segmentation_model(use_quantization=False):
    """
    Create semantic segmentation model
    
    Args:
        use_quantization: Whether to use QKeras quantized layers
    """
    tf.keras.backend.clear_session()
    
    # Input
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    
    # Initial feature extraction
    x = conv_block(x, MODEL_CONFIG['init_filters'], use_quantization=use_quantization, name='init_conv')
    
    # Encoder path with skip connections
    skip_connections = []
    
    # Encoder block 1
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], strides=2, use_quantization=use_quantization, name='enc1_down')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, name='enc1_conv')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.2, name='dropout_encoder1')(x)
    skip_connections.append(x)
    
    # Encoder block 2
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], strides=2, use_quantization=use_quantization, name='enc2_down')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, name='enc2_conv')
    x = attention_block(x, use_quantization=use_quantization, name='enc2_att')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.25, name='dropout_encoder2')(x)
    skip_connections.append(x)
    
    # Encoder block 3
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], strides=2, use_quantization=use_quantization, name='enc3_down')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, name='enc3_conv')
    x = attention_block(x, use_quantization=use_quantization, name='enc3_att')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.3, name='dropout_encoder3')(x)
    skip_connections.append(x)
    
    # Bridge block
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], strides=2, use_quantization=use_quantization, name='bridge_down')
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], use_quantization=use_quantization, name='bridge_conv1')
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], use_quantization=use_quantization, name='bridge_conv2')
    x = attention_block(x, use_quantization=use_quantization, name='bridge_att')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.35, name='dropout_bridge')(x)
    
    # Decoder path
    # Decoder block 3
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec3_up')(x)
    x = tf.keras.layers.Concatenate(name='dec3_concat')([x, skip_connections[2]])
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, name='dec3_conv1')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, name='dec3_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.25, name='dropout_decoder3')(x)
    
    # Decoder block 2
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec2_up')(x)
    x = tf.keras.layers.Concatenate(name='dec2_concat')([x, skip_connections[1]])
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, name='dec2_conv1')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, name='dec2_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.2, name='dropout_decoder2')(x)
    
    # Decoder block 1
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec1_up')(x)
    x = tf.keras.layers.Concatenate(name='dec1_concat')([x, skip_connections[0]])
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, name='dec1_conv1')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, name='dec1_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.15, name='dropout_decoder1')(x)
    
    # Final upsampling to original resolution
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='final_up')(x)
    
    # Final convolution before output
    x = conv_block(x, MODEL_CONFIG['init_filters'], use_quantization=use_quantization, name='final_conv')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.1, name='dropout_final')(x)
    
    # Output layer
    if use_quantization:
        outputs = QConv2D(NUM_CLASSES, kernel_size=1, padding='same',
                         kernel_quantizer=quantized_bits(**quantized_bits_config),
                         bias_quantizer=quantized_bits(**quantized_bits_config),
                         activation='softmax', name='output_qconv')(x)
    else:
        outputs = Conv2D(NUM_CLASSES, kernel_size=1, padding='same',
                        activation='softmax', name='output_conv')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='segmentation_model')
    return model

def create_hls4ml_export_model(trained_model=None, use_quantization=False):
    """Create a clean model for HLS4ML export without dropout layers"""
    # Temporarily disable dropout
    global USE_DROPOUT_DURING_TRAINING
    original_setting = USE_DROPOUT_DURING_TRAINING
    USE_DROPOUT_DURING_TRAINING = False
    
    # Create clean model without dropout
    clean_model = create_segmentation_model(use_quantization=use_quantization)
    
    # Restore original setting
    USE_DROPOUT_DURING_TRAINING = original_setting
    
    # If a trained model is provided, copy weights by layer name
    if trained_model is not None:
        print("Copying weights from trained model to clean model...")
        
        # Create a mapping of layer names to trained layers
        trained_layers_dict = {layer.name: layer for layer in trained_model.layers}
        
        # Copy weights by matching layer names
        copied_count = 0
        for clean_layer in clean_model.layers:
            if clean_layer.name in trained_layers_dict:
                trained_layer = trained_layers_dict[clean_layer.name]
                
                # Check if both layers have weights and they're compatible
                if (hasattr(clean_layer, 'weights') and hasattr(trained_layer, 'weights') and
                    len(clean_layer.weights) > 0 and len(trained_layer.weights) > 0):
                    
                    try:
                        # Check if weight shapes are compatible
                        clean_weights = clean_layer.get_weights()
                        trained_weights = trained_layer.get_weights()
                        
                        if len(clean_weights) == len(trained_weights):
                            # Check if all weight shapes match
                            shapes_compatible = True
                            for cw, tw in zip(clean_weights, trained_weights):
                                if cw.shape != tw.shape:
                                    shapes_compatible = False
                                    break
                            
                            if shapes_compatible:
                                clean_layer.set_weights(trained_weights)
                                copied_count += 1
                            else:
                                print(f"Skipping {clean_layer.name}: weight shapes incompatible")
                        else:
                            print(f"Skipping {clean_layer.name}: different number of weights")
                    except Exception as e:
                        print(f"Error copying weights for {clean_layer.name}: {str(e)}")
        
        print(f"Successfully copied weights for {copied_count} layers")
    
    print("Clean model created for HLS4ML export (no dropout layers)")
    return clean_model

def create_data_generators():
    """Create training and validation data generators"""
    train_image_dir = os.path.join(DATASET_DIR, 'leftImg8bit', 'train')
    val_image_dir = os.path.join(DATASET_DIR, 'leftImg8bit', 'val')
    train_mask_dir = os.path.join(DATASET_DIR, 'gtFine', 'train')
    val_mask_dir = os.path.join(DATASET_DIR, 'gtFine', 'val')
    
    train_generator = CityscapesDataGenerator(
        train_image_dir, train_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=True
    )
    
    val_generator = CityscapesDataGenerator(
        val_image_dir, val_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=False,
        augment=False
    )
    
    return train_generator, val_generator

def convert_float32_to_qat(float32_model):
    """
    Convert a trained float32 model to QAT model by replacing layers
    This function creates a new QAT model and transfers weights from float32 model
    """
    print("Converting float32 model to QAT model...")
    
    # Create new QAT model with same architecture
    qat_model = create_segmentation_model(use_quantization=True)
    
    # Transfer weights from float32 to QAT model where possible
    float32_layers = {layer.name: layer for layer in float32_model.layers}
    
    for qat_layer in qat_model.layers:
        layer_name = qat_layer.name
        
        # Map QAT layer names to float32 layer names
        if layer_name.endswith('_qconv'):
            float32_name = layer_name.replace('_qconv', '_conv')
        elif layer_name.endswith('_qbn'):
            float32_name = layer_name.replace('_qbn', '_bn')
        else:
            float32_name = layer_name
        
        if float32_name in float32_layers:
            float32_layer = float32_layers[float32_name]
            
            try:
                # Transfer weights for compatible layers
                if hasattr(qat_layer, 'set_weights') and hasattr(float32_layer, 'get_weights'):
                    weights = float32_layer.get_weights()
                    if len(weights) > 0:
                        qat_layer.set_weights(weights)
                        print(f"Transferred weights: {float32_name} -> {layer_name}")
            except Exception as e:
                print(f"Could not transfer weights for {layer_name}: {e}")
    
    return qat_model

def train_model():
    """Main training function"""
    print(f"Starting training with QAT={'Enabled' if TRAIN_QAT else 'Disabled'}")
    print(f"Memory usage at start: {get_memory_usage():.2f} MB")
    
    # Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Create models directory if it doesn't exist
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Check for existing models to resume training automatically
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
    initial_epoch = 0
    skip_compilation = False
    model = None
    model_prefix = "qat_model" if TRAIN_QAT else "float32_model"
    
    # Filter model files by training type (float32 vs QAT)
    if TRAIN_QAT:
        # Look for QAT models when in QAT mode
        relevant_models = [f for f in model_files if 'qat_model' in f and 'checkpoint' in f]
    else:
        # Look for float32 models when in float32 mode
        relevant_models = [f for f in model_files if 'float32_model' in f and 'checkpoint' in f]
    
    if relevant_models and not LOAD_MODEL_PATH:
        # Sort by modification time to get the latest
        latest_model = max(relevant_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        print(f"\nFound latest checkpoint: {latest_model}")
        
        # Extract epoch number if it exists in filename
        if 'epoch_' in latest_model:
            try:
                if 'checkpoint_epoch_' in latest_model:
                    epoch_part = latest_model.split('checkpoint_epoch_')[1]
                else:
                    epoch_part = latest_model.split('epoch_')[1]
                
                # Extract epoch number from filename
                initial_epoch = int(epoch_part.split('.')[0])
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not extract epoch number, starting from epoch 0")
                initial_epoch = 0
        
        # Try to load as full model first, then as weights only
        try:
            print(f"Trying to load full model from {latest_model}")
            model = tf.keras.models.load_model(os.path.join(model_dir, latest_model),
                                               custom_objects={
                                                   'QConv2D': QConv2D,
                                                   'QBatchNormalization': QBatchNormalization,
                                                   'QActivation': QActivation,
                                                   'quantized_bits': quantized_bits,
                                                   'quantized_relu': quantized_relu,
                                                   'ComboLoss': ComboLoss
                                               })
            print(f"Successfully loaded model from {latest_model}")
            skip_compilation = True  # Model already compiled
            
        except Exception as e_model:
            print(f"Failed to load full model: {e_model}")
            try:
                print(f"Trying to load weights from {latest_model}")
                # Create new model and load weights
                if TRAIN_QAT:
                    model = create_segmentation_model(use_quantization=True)
                else:
                    model = create_segmentation_model(use_quantization=False)
                model.load_weights(os.path.join(model_dir, latest_model))
                print(f"Successfully loaded weights from {latest_model}")
                initial_epoch = 0  # Reset epoch if only weights loaded
            except Exception as e_weights:
                print(f"Failed to load weights: {e_weights}")
                print("Creating new model")
                model = None
                initial_epoch = 0
    
    # Create or load model if not already loaded
    if model is None:
        if TRAIN_QAT and LOAD_MODEL_PATH:
            print(f"Loading float32 model from: {LOAD_MODEL_PATH}")
            float32_model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
            model = convert_float32_to_qat(float32_model)
        elif TRAIN_QAT:
            print("Creating QAT model from scratch")
            model = create_segmentation_model(use_quantization=True)
        else:
            print("Creating float32 baseline model")
            model = create_segmentation_model(use_quantization=False)
    
    # Compile model (only if not already compiled)
    if not skip_compilation:
        if TRAIN_QAT:
            # Use lower learning rate for QAT fine-tuning
            learning_rate = 1e-5
        else:
            # Standard learning rate for float32 training
            learning_rate = 1e-3
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=ComboLoss(alpha_iou=0.3, alpha_ce=0.7),  # Simple constant weights: 30% IoU, 70% CrossEntropy
            metrics=[tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, name='mean_iou')]
        )
        print(f"Model compiled with learning rate: {learning_rate}")
    else:
        print("Skipping compilation - using loaded model with existing optimizer state")
    
    model.summary()
    
    # Create callbacks
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_checkpoint_epoch_{{epoch:02d}}.keras'),
            save_weights_only=False,
            save_best_only=False,
            monitor='val_mean_iou',
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_best.keras'),
            save_weights_only=False,
            save_best_only=True,
            monitor='val_mean_iou',
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_mean_iou',
            mode='max',
            factor=0.7,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_mean_iou',
            mode='max',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        MemoryCleanupCallback(),
        TrainingLoggerCallback(model_dir)
    ]
    
    # Train model
    try:
        print(f"Starting training from epoch {initial_epoch}...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(model_dir, f'{model_prefix}_final.keras')
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Create and save clean model for HLS4ML export (without dropout)
        print("Creating clean model for HLS4ML export...")
        clean_model = create_hls4ml_export_model(trained_model=model, use_quantization=TRAIN_QAT)
        clean_model_path = os.path.join(model_dir, f'{model_prefix}_clean_for_hls4ml.keras')
        clean_model.save(clean_model_path)
        print(f"Clean model (no dropout) saved to: {clean_model_path}")
        
        # If QAT training, save quantized weights for hls4ml
        if TRAIN_QAT:
            quantized_weights_path = os.path.join(model_dir, f'{model_prefix}_quantized_weights.h5')
            model_save_quantized_weights(clean_model, quantized_weights_path)  # Use clean model for quantized weights
            print(f"Quantized weights saved to: {quantized_weights_path}")
            print("Note: Quantized weights saved from clean model (without dropout) for HLS4ML compatibility")
        
        print("Training completed successfully!")
        
        # Print training summary
        if hasattr(history, 'history'):
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_miou = history.history['mean_iou'][-1]
            final_val_miou = history.history['val_mean_iou'][-1]
            
            print(f"\nFinal Training Results:")
            print(f"  Training Loss: {final_loss:.4f}")
            print(f"  Validation Loss: {final_val_loss:.4f}")
            print(f"  Training Mean IoU: {final_miou:.4f}")
            print(f"  Validation Mean IoU: {final_val_miou:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupt_model_path = os.path.join(model_dir, f'{model_prefix}_interrupted.keras')
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        error_model_path = os.path.join(model_dir, f'{model_prefix}_error.keras')
        model.save(error_model_path)
        print(f"Model saved to: {error_model_path}")
        raise
    finally:
        cleanup_memory()
        print(f"Final memory usage: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    print("="*60)
    print("HLS4ML Semantic Segmentation Training")
    print("="*60)
    
    if not TRAIN_QAT:
        print("Phase 1: Training float32 baseline model")
        print("This will create a high-quality baseline model without quantization")
        print("After training completes, use --trainQAT=True to apply QAT")
    else:
        print("Phase 2: Applying Quantization-Aware Training (QAT)")
        print("This will fine-tune the model with quantized layers for FPGA deployment")
        if LOAD_MODEL_PATH:
            print(f"Loading baseline model from: {LOAD_MODEL_PATH}")
        else:
            print("Training QAT model from scratch (not recommended)")
    
    print("="*60)
    
    train_model()
