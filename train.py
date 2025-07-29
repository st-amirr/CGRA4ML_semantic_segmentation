import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization
from qkeras import QActivation, QConv2D, QConv2DTranspose, QBatchNormalization, quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path
import math

# python3 /app/train.py

# IMPROVEMENTS FOR OVERFITTING PREVENTION (HLS4ML-COMPATIBLE):
# 1. Lightweight Data Augmentation (optimized for speed):
#    - Essential geometric: horizontal flip, rotation, shift-scale-rotate (reduced limits)
#    - Lightweight color: brightness/contrast, gamma, hue/sat/value (reduced ranges)
#    - Minimal noise: Gaussian noise (reduced variance)
#    - Simple cutout: CoarseDropout (reduced size and probability)
#    - Removed heavy augmentations: crops, blur, elastic, weather effects
# 2. Smart Regularization Strategy:
#    - Added L2 regularization (1e-4) to all QConv2D layers (inference-compatible)
#    - Added Dropout layers during training (controlled by USE_DROPOUT_DURING_TRAINING toggle)
#    - Automatic creation of clean model for HLS4ML export (without dropout)
#    - Model capacity maintained (no reduction for performance)
# 3. Learning Rate and Scheduling (maintained):
#    - Reduced initial LR from 1e-4 to 5e-5
#    - More aggressive LR reduction (factor: 0.3->0.2, patience: 3->2)
#    - Reduced early stopping patience from 10 to 7 epochs
# 4. Data Validation (maintained):
#    - Added validation function to check for data leakage
#    - Ensures proper train/val split and matching counts

# Define constants
INPUT_HEIGHT = 1024
INPUT_WIDTH = 2048
BATCH_SIZE = 1  # Keep at 1 for memory constraints
EPOCHS = 50
NUM_CLASSES = 34
DATASET_DIR = '/cityscapes'
GRADIENT_ACCUMULATION_STEPS = 8  # Keep at 1 for now
NUM_WORKERS = 1  # Set to 1 for stability and minimal memory usage
PREFETCH_SIZE = 2  # Reduced for stability

# Toggle for dropout usage (True for training, False for HLS4ML export)
USE_DROPOUT_DURING_TRAINING = True

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=20000)]
            )
            
        # Disable layout optimizer and other problematic optimizations
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': False,  # Disable layout optimization
            'constant_folding': True,
            'shape_optimization': False,  # Disable shape optimization
            'remapping': False,  # Disable remapping
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'scoped_allocator_optimization': False,  # Disable scoped allocator
            'implementation_selector': False,  # Disable implementation selector
        })
        
        print("GPU memory growth enabled with optimizations configured for QAT")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")

# Configure TensorFlow to use CPU efficiently
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)

# Enable memory growth and tensor allocation reporting
tf.config.set_soft_device_placement(True)

# Disable eager execution for better memory tracking
# tf.compat.v1.disable_eager_execution()  # Commented out to enable eager execution

# Add a check to confirm eager execution is enabled
print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Add TF memory logger
tf.debugging.set_log_device_placement(False)  # Set to True only for debugging

# Define quantization parameters optimized for FPGA implementation
quantized_bits_param = {
    'bits': 8,
    'integer': 4,
    'symmetric': True,  # For better HLS4ML compatibility
    'alpha': 'auto_po2',      # Use automatic power-of-two scaling
    'use_stochastic_rounding': True,  # Enable for better training stability
    'scale_axis': None
}

quantized_relu_param = {
    'bits': 8,
    'integer': 4,
    'use_stochastic_rounding': True  # Enable for better training stability
}

# Update quantized_bits_param for 8-bit layers
quantized_bits_param_8 = {
    'bits': 8,
    'integer': 4,
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}
# For first/last layers, use 16 bits
quantized_bits_param_16 = {
    'bits': 16,
    'integer': 8,
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# Update quantized_relu_param for 8-bit layers
quantized_relu_param_8 = {
    'bits': 8,
    'integer': 4,
    'use_stochastic_rounding': True
}
# For first/last layers, use 16 bits
quantized_relu_param_16 = {
    'bits': 16,
    'integer': 8,
    'use_stochastic_rounding': True
}

def conv_block(x, filters, kernel_size=3, strides=1, name=None, use_16bit=False):
    """FPGA-optimized convolution block with QAT"""
    qbits = quantized_bits_param_16 if use_16bit else quantized_bits_param_8
    qrelu = quantized_relu_param_16 if use_16bit else quantized_relu_param_8
    x = QConv2D(filters, kernel_size, strides=strides, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_conv')(x)
    x = QBatchNormalization(
        gamma_quantizer=quantized_bits(**qbits),
        beta_quantizer=quantized_bits(**qbits),
        mean_quantizer=quantized_bits(**qbits),
        variance_quantizer=quantized_bits(**qbits),
        momentum=0.9,
        epsilon=1e-5,
        name=f'{name}_bn'
    )(x)
    x = QActivation(quantized_relu(**qrelu), name=f'{name}_act')(x)
    return x

def c2f_block(x, filters, n=1, name=None):
    """YOLOv8 C2f block with quantization optimized for VCK5000"""
    # First conv
    x = conv_block(x, filters, name=f'{name}_conv1')
    
    # Split
    y = x
    
    # Bottleneck blocks - optimized for VCK5000
    for i in range(n):
        # Use 1x1 conv for channel reduction
        y = conv_block(y, filters//2, kernel_size=1, name=f'{name}_bottleneck{i}_conv1')
        y = conv_block(y, filters, kernel_size=3, name=f'{name}_bottleneck{i}_conv2')
        x = tf.keras.layers.Concatenate(name=f'{name}_concat{i}')([x, y])
    
    return x

# Add SPPF module for better multi-scale feature extraction
def sppf_block(x, channels, name):
    """Spatial Pyramid Pooling - Fast (SPPF) with quantization"""
    y1 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5')(x)
    y2 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5_2')(y1)
    y3 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5_3')(y2)
    
    concat = tf.keras.layers.Concatenate(name=f'{name}_concat')([x, y1, y2, y3])
    
    return conv_block(concat, channels, kernel_size=1, name=f'{name}_conv')

# Efficient Channel Attention Module
def channel_attention(x, ratio=16, name=''):
    """Lightweight Squeeze-and-Excitation block with quantization"""
    channels = x.shape[-1]
    
    # Global average pooling
    y = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    y = tf.keras.layers.Reshape((1, 1, channels), name=f'{name}_reshape')(y)
    
    # Single conv for efficiency
    y = QConv2D(channels, 1,
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_attention')(y)
    y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    
    return tf.keras.layers.Multiply(name=f'{name}_scale')([x, y])

def create_segmentation_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    # Initial conv (use 16 bits) - back to original 32
    x = conv_block(x, 32, name='init_conv', use_16bit=True)
    x = QConv2D(32, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample1')(x)
    # Stage 1 - back to original 64
    x = c2f_block(x, 64, n=2, name='stage1')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_stage1')(x)
    skip1 = x
    x = QConv2D(64, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample2')(x)
    # Stage 2 - back to original 128
    x = c2f_block(x, 128, n=2, name='stage2')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_stage2')(x)
    skip2 = x
    x = QConv2D(128, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample3')(x)
    # Stage 3 - back to original 256
    x = c2f_block(x, 256, n=1, name='stage3')
    x = channel_attention(x, name='ca_stage3')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.3, name='dropout_stage3')(x)
    skip3 = x
    x = QConv2D(256, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample4')(x)
    # Stage 4 - back to original 512
    x = c2f_block(x, 512, n=1, name='stage4')
    x = channel_attention(x, name='ca_stage4')
    x = sppf_block(x, 512, name='sppf')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.35, name='dropout_stage4')(x)
    # Decoder (upsample and match filter counts)
    x = c2f_block(x, 512, n=1, name='pre_up1')
    x = QConv2DTranspose(256, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                        bias_quantizer=quantized_bits(**quantized_bits_param_8),
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        name='up1_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat1')([x, skip3])
    x = conv_block(x, 256, name='up1_conv')
    x = c2f_block(x, 256, n=1, name='up1_c2f')
    x = channel_attention(x, name='ca_up1')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_up1')(x)
    x = c2f_block(x, 256, n=1, name='pre_up2')
    x = QConv2DTranspose(128, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                        bias_quantizer=quantized_bits(**quantized_bits_param_8),
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        name='up2_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat2')([x, skip2])
    x = conv_block(x, 128, name='up2_conv')
    x = c2f_block(x, 128, n=1, name='up2_c2f')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_up2')(x)
    x = c2f_block(x, 128, n=1, name='pre_up3')
    x = QConv2DTranspose(64, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                        bias_quantizer=quantized_bits(**quantized_bits_param_8),
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        name='up3_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat3')([x, skip1])
    x = conv_block(x, 64, name='up3_conv')
    x = c2f_block(x, 64, n=1, name='up3_c2f')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.15, name='dropout_up3')(x)
    x = c2f_block(x, 64, n=1, name='pre_up4')
    x = QConv2DTranspose(32, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                        bias_quantizer=quantized_bits(**quantized_bits_param_8),
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        name='up4_transpose')(x)
    x = conv_block(x, 32, name='up4_conv')
    x = c2f_block(x, 32, n=1, name='up4_c2f')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.1, name='dropout_up4')(x)
    # Output layer (use 16 bits)
    outputs = QConv2D(NUM_CLASSES, (1, 1),
                     kernel_quantizer=quantized_bits(**quantized_bits_param_16),
                     bias_quantizer=quantized_bits(**quantized_bits_param_16),
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                     name='output')(x)
    outputs = QActivation(quantized_bits(**quantized_bits_param_16), name='output_quant')(outputs)
    outputs = tf.keras.layers.Activation('softmax', name='output_softmax')(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_hls4ml_export_model(trained_model=None):
    """Create a clean model for HLS4ML export without dropout layers"""
    # Temporarily disable dropout
    global USE_DROPOUT_DURING_TRAINING
    original_setting = USE_DROPOUT_DURING_TRAINING
    USE_DROPOUT_DURING_TRAINING = False
    
    # Create clean model without dropout
    clean_model = create_segmentation_model()
    
    # Restore original setting
    USE_DROPOUT_DURING_TRAINING = original_setting
    
    # If a trained model is provided, copy weights (excluding dropout layers)
    if trained_model is not None:
        print("Copying weights from trained model to clean model...")
        for clean_layer, trained_layer in zip(clean_model.layers, trained_model.layers):
            if hasattr(clean_layer, 'weights') and hasattr(trained_layer, 'weights'):
                if len(clean_layer.weights) > 0 and len(trained_layer.weights) > 0:
                    clean_layer.set_weights(trained_layer.get_weights())
                    print(f"Copied weights for layer: {clean_layer.name}")
    
    print("Clean model created for HLS4ML export (no dropout layers)")
    return clean_model

class CityscapesDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, augment=False, preprocessed_prefix=None):
        self.batch_size = batch_size
        self.augment = augment
        self.indices = None
        self.preprocessed = False
        self.images = None
        self.masks = None
        self.target_size = (INPUT_WIDTH, INPUT_HEIGHT)
        self.aug = None
        if augment:
            self.aug = A.Compose([
                # Essential geometric transformations (lightweight)
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # Reduced from 0.1
                    scale_limit=0.1,   # Reduced from 0.2
                    rotate_limit=5,    # Reduced from 10
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                
                # Lightweight color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # Reduced from 0.3
                    contrast_limit=0.2,    # Reduced from 0.3
                    p=0.3
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),  # Reduced range
                A.HueSaturationValue(
                    hue_shift_limit=10,    # Reduced from 20
                    sat_shift_limit=15,    # Reduced from 30
                    val_shift_limit=10,    # Reduced from 20
                    p=0.2
                ),
                
                # Lightweight noise (minimal impact on speed)
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),  # Reduced variance
                
                # Simple cutout for regularization (efficient)
                A.CoarseDropout(
                    max_holes=4,      # Reduced from 8
                    max_height=32,    # Reduced from 64
                    max_width=32,     # Reduced from 64
                    min_holes=1,
                    min_height=8,     # Reduced from 16
                    min_width=8,      # Reduced from 16
                    fill_value=0,
                    p=0.15           # Reduced probability
                ),
            ])
        # Try to load preprocessed data if prefix is given
        if preprocessed_prefix is not None:
            img_file = f"{preprocessed_prefix}_images.npy"
            mask_file = f"{preprocessed_prefix}_masks.npy"
            if os.path.exists(img_file) and os.path.exists(mask_file):
                print(f"Loading preprocessed data from {img_file} and {mask_file}")
                self.images = np.load(img_file)
                self.masks = np.load(mask_file)
                self.indices = np.arange(len(self.images))
                self.preprocessed = True
            else:
                print(f"Preprocessed files not found for prefix {preprocessed_prefix}, falling back to raw images.")
        if not self.preprocessed:
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        if self.preprocessed:
            return len(self.images) // self.batch_size
        return len(self.image_paths) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        gc.collect()
        # tf.keras.backend.clear_session()  # Removed to prevent training hang when resuming
    
    def load_and_preprocess_image(self, img_path, mask_path):
        try:
            # Load and preprocess image using cv2 for faster loading
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Use cv2 for faster resizing
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            
            # Load and preprocess mask using cv2 for faster loading
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to one-hot encoding using numpy for better performance
            mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
            np.put_along_axis(mask_one_hot, mask[..., np.newaxis], 1, axis=-1)
            
            if self.augment and self.aug is not None:
                augmented = self.aug(image=img, mask=mask_one_hot)
                img = augmented['image']
                mask_one_hot = augmented['mask']
                
                # Ensure correct dimensions after augmentation
                if img.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                if mask_one_hot.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                    mask_one_hot = cv2.resize(mask_one_hot, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            return img, mask_one_hot
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.preprocessed:
            X = self.images[batch_indices]
            Y = self.masks[batch_indices]
            # Optionally add augmentation here if needed
            return X, Y
        else:
            batch_x = [self.image_paths[i] for i in batch_indices]
            batch_y = [self.mask_paths[i] for i in batch_indices]
            X = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
            Y = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
            for i, (img_path, mask_path) in enumerate(zip(batch_x, batch_y)):
                img, mask = self.load_and_preprocess_image(img_path, mask_path)
                if img is not None and mask is not None:
                    X[i] = img
                    Y[i] = mask
            return X, Y

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Clean up memory at the end of each epoch"""
        gc.collect()
        tf.keras.backend.clear_session()
        # Log memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"\nMemory usage after epoch {epoch}: {memory_usage:.2f} MB")

class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, clr_step=10, checkpoint_step=30):
        super().__init__()
        self.log_file = os.path.join(log_dir, 'log.txt')
        os.makedirs(log_dir, exist_ok=True)
        self.clr_step = clr_step
        self.checkpoint_step = checkpoint_step
        self.batch_count = 0
        # Ensure the log file exists
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    f.write('')
            except Exception as e:
                print(f"[TrainingLoggerCallback] Could not create log.txt: {e}")

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_count += 1
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        log_line = f"{self.batch_count}/{{steps}} [batch] - loss: {logs.get('loss', float('nan')):.4f} - iou: {logs.get('iou', float('nan')):.4f} - lr: {lr:.6f}\n"
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"[TrainingLoggerCallback] Could not write to log.txt: {e}")
        # Log CLR change
        if self.clr_step and self.batch_count % self.clr_step == 0:
            clr_line = f"Batch {self.batch_count}: CyclicalLearningRate set learning rate to {lr}\n"
            try:
                with open(self.log_file, 'a') as f:
                    f.write(clr_line)
            except Exception as e:
                print(f"[TrainingLoggerCallback] Could not write CLR to log.txt: {e}")
        # Log checkpoint save
        if self.checkpoint_step and self.batch_count % self.checkpoint_step == 0:
            checkpoint_line = f"Batch {self.batch_count}: saving model to /app/models/checkpoint_epoch_{{epoch:02d}}_batch_{{batch:04d}}.h5\n"
            try:
                with open(self.log_file, 'a') as f:
                    f.write(checkpoint_line)
            except Exception as e:
                print(f"[TrainingLoggerCallback] Could not write checkpoint to log.txt: {e}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_line = f"Epoch {epoch+1}: "
        log_line += ", ".join([f"{k}={v:.5f}" for k, v in logs.items() if isinstance(v, (float, int))])
        log_line += f"\nEpoch {epoch+1}: saving model to /app/models/checkpoint_epoch_{epoch+1:02d}_batch_0000.h5\n"
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"[TrainingLoggerCallback] Could not write to log.txt: {e}")

# Custom IoU metric for semantic segmentation
class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=NUM_CLASSES, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.intersection = self.add_weight(name='intersection', shape=(num_classes,), initializer='zeros')
        self.union = self.add_weight(name='union', shape=(num_classes,), initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Ensure both tensors have the same shape and dtype
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        
        # Calculate intersection and union for each class
        intersections = []
        unions = []
        
        for i in range(self.num_classes):
            # Create binary masks for current class
            pred_mask = tf.cast(tf.equal(y_pred, i), tf.float32)
            true_mask = tf.cast(tf.equal(y_true, i), tf.float32)
            
            # Calculate intersection and union
            intersection = tf.reduce_sum(pred_mask * true_mask)
            union = tf.reduce_sum(pred_mask) + tf.reduce_sum(true_mask) - intersection
            
            intersections.append(intersection)
            unions.append(union)
        
        # Update state variables using scatter_nd_update
        indices = tf.range(self.num_classes, dtype=tf.int32)
        indices = tf.reshape(indices, [-1, 1])
        
        self.intersection.assign_add(tf.scatter_nd(indices, tf.stack(intersections), [self.num_classes]))
        self.union.assign_add(tf.scatter_nd(indices, tf.stack(unions), [self.num_classes]))

    def result(self):
        # Calculate IoU for each class
        iou = tf.math.divide_no_nan(self.intersection, self.union)
        # Return mean IoU across all classes
        return tf.reduce_mean(iou)

    def reset_state(self):
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.union.assign(tf.zeros_like(self.union))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'num_classes': self.num_classes}

def validate_data_split(train_images, train_masks, val_images, val_masks):
    """Validate data split to ensure no data leakage and proper matching"""
    print("\n=== Data Validation ===")
    
    # Check for data leakage (overlap between train and val)
    train_image_paths = set(str(img) for img in train_images)
    val_image_paths = set(str(img) for img in val_images)
    
    overlap = train_image_paths.intersection(val_image_paths)
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping images between train and val sets!")
        print("This indicates data leakage. Please check your data split.")
        for img in list(overlap)[:5]:  # Show first 5 examples
            print(f"  - {img}")
    else:
        print("✓ No data leakage detected - train and val sets are properly separated")
    
    # Check for matching counts
    print(f"✓ Training images: {len(train_images)}")
    print(f"✓ Training masks: {len(train_masks)}")
    print(f"✓ Validation images: {len(val_images)}")
    print(f"✓ Validation masks: {len(val_masks)}")
    
    if len(train_images) != len(train_masks):
        print(f"ERROR: Mismatch in training data - {len(train_images)} images vs {len(train_masks)} masks")
    else:
        print("✓ Training image/mask counts match")
    
    if len(val_images) != len(val_masks):
        print(f"ERROR: Mismatch in validation data - {len(val_images)} images vs {len(val_masks)} masks")
    else:
        print("✓ Validation image/mask counts match")
    
    # Check for empty sets
    if not train_images:
        raise ValueError("No training images found!")
    if not val_images:
        raise ValueError("No validation images found!")
    
    print("=== Data Validation Complete ===\n")

def prepare_data(cityscapes_root):
    # Assuming standard Cityscapes directory structure
    images_dir = Path(cityscapes_root) / 'leftImg8bit'
    labels_dir = Path(cityscapes_root) / 'gtFine'
    
    print(f"Looking for images in: {images_dir}")
    print(f"Looking for labels in: {labels_dir}")
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    train_images = sorted(list(images_dir.glob('train/*/*.png')))
    train_masks = sorted(list(labels_dir.glob('train/*/*_labelIds.png')))
    val_images = sorted(list(images_dir.glob('val/*/*.png')))
    val_masks = sorted(list(labels_dir.glob('val/*/*_labelIds.png')))
    
    print("\nFound directories:")
    print(f"Train images directory: {list(images_dir.glob('train'))}")
    print(f"Train labels directory: {list(labels_dir.glob('train'))}")
    print(f"Val images directory: {list(images_dir.glob('val'))}")
    print(f"Val labels directory: {list(labels_dir.glob('val'))}")
    
    if not train_images:
        raise ValueError(f"No training images found in {images_dir}/train/*/*.png")
    if not train_masks:
        raise ValueError(f"No training masks found in {labels_dir}/train/*/*_labelIds.png")
    if not val_images:
        raise ValueError(f"No validation images found in {images_dir}/val/*/*.png")
    if not val_masks:
        raise ValueError(f"No validation masks found in {labels_dir}/val/*/*_labelIds.png")
    
    return train_images, train_masks, val_images, val_masks

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CyclicalLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=1e-4, max_lr=5e-3, step_size=10, mode='triangular', verbose=1):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.verbose = verbose
        self.iteration = 0
        self.history = {}

    def clr(self):
        cycle = math.floor(1 + self.iteration / (2 * self.step_size))
        x = abs(self.iteration / self.step_size - 2 * cycle + 1)
        scale = 1.0
        if self.mode == 'triangular2':
            scale = 1 / (2. ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = 0.99994 ** self.iteration
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale

    def on_train_begin(self, logs=None):
        self.iteration = 0
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault('lr', []).append(lr)
        if self.verbose and self.iteration % self.step_size == 0:
            print(f"\nBatch {self.iteration}: CyclicalLearningRate set learning rate to {lr}")

def train_model():
    # Create models directory if it doesn't exist
    os.makedirs('/app/models', exist_ok=True)
    
    # Find the latest saved model
    model_files = [f for f in os.listdir('/app/models') if f.endswith('.h5')]
    initial_epoch = 0
    skip_compilation = False  # Initialize compilation flag
    
    if model_files:
        # Sort by modification time to get the latest
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('/app/models', x)))
        print(f"\nFound latest model: {latest_model}")
        
        # Extract epoch number if it exists in filename
        if 'epoch_' in latest_model:
            try:
                initial_epoch = int(latest_model.split('epoch_')[1].split('_')[0])
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not determine epoch number from filename")
        
        # Try to load as full model first, then as weights only
        model = None
        try:
            print(f"Trying to load full model from {latest_model}")
            model = tf.keras.models.load_model('/app/models/' + latest_model,
                                           custom_objects={
                                               'QConv2D': QConv2D,
                                               'QConv2DTranspose': QConv2DTranspose,
                                               'QBatchNormalization': QBatchNormalization,
                                               'QActivation': QActivation,
                                               'IoUMetric': IoUMetric,
                                               'quantized_bits': quantized_bits,
                                               'quantized_relu': quantized_relu
                                           })
            print("Successfully loaded full model with optimizer state.")
            # Don't recompile - the loaded model already has optimizer state
            skip_compilation = True
        except Exception as e_model:
            print(f"Failed to load as full model: {str(e_model)}")
            try:
                print(f"Trying to load weights from {latest_model}")
                model = create_segmentation_model()
                model.load_weights('/app/models/' + latest_model)
                print("Successfully loaded weights into model architecture.")
                skip_compilation = False  # Need to compile since we only loaded weights
            except Exception as e_weights:
                print(f"Error loading model: {str(e_weights)}")
                print("Will start training from scratch")
                model = create_segmentation_model()
                initial_epoch = 0
                skip_compilation = False
    else:
        print("\nNo existing models found. Creating new model.")
        model = create_segmentation_model()
        initial_epoch = 0
    
    # Compile with QAT-specific settings and disable XLA (only if not already compiled)
    if not skip_compilation:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5,  # Reduced from 1e-4 for better stability
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[IoUMetric()],
            run_eagerly=False,
            # Remove jit_compile if it was added
        )
        print("Model compiled successfully.")
    else:
        print("Skipping compilation - using loaded model with existing optimizer state.")
    
    # Setup data paths
    cityscapes_root = '/data'
    print(f"\nLooking for Cityscapes dataset in: {cityscapes_root}")
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    # Validate data split to prevent data leakage
    validate_data_split(train_images, train_masks, val_images, val_masks)
    
    print(f"\nFound {len(train_images)} training images and {len(val_images)} validation images")
    
    # Try to use preprocessed data if available
    train_imgs_npy = os.path.join('models', 'train_images.npy')
    train_msks_npy = os.path.join('models', 'train_masks.npy')
    val_imgs_npy = os.path.join('models', 'val_images.npy')
    val_msks_npy = os.path.join('models', 'val_masks.npy')
    use_preprocessed = all(os.path.exists(f) for f in [train_imgs_npy, train_msks_npy, val_imgs_npy, val_msks_npy])
    if use_preprocessed:
        train_gen = CityscapesDataGenerator(None, None, BATCH_SIZE, augment=True, preprocessed_prefix=os.path.join('models', 'train'))
        val_gen = CityscapesDataGenerator(None, None, BATCH_SIZE, augment=False, preprocessed_prefix=os.path.join('models', 'val'))
    else:
        train_gen = CityscapesDataGenerator(train_images, train_masks, BATCH_SIZE, augment=True)
        val_gen = CityscapesDataGenerator(val_images, val_masks, BATCH_SIZE, augment=False)

    # Remove tf.data.Dataset.from_generator and prefetch
    # Use generators directly in model.fit

    # Remove HLS4ML model saving callback
    # class HLS4MLModelSaver(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         if (epoch + 1) % 10 == 0:  # Save every 10 epochs
    #             # Save quantized weights in HLS4ML compatible format
    #             model_save_quantized_weights(
    #                 self.model,
    #                 f'/app/models/hls4ml_weights_epoch_{epoch+1}.h5'
    #             )
    
    callbacks = [
        ModelCheckpoint(
            '/app/models/best_model.h5',
            save_best_only=True,
            monitor='val_iou',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        # Save full model every 30 batches (preserves optimizer state)
        ModelCheckpoint(
            '/app/models/checkpoint_epoch_{epoch:02d}_batch_{batch:04d}.h5',
            save_freq=30,
            verbose=1,
            save_weights_only=False
        ),
        # Save full model every epoch (batch always 0000 for epoch save)
        ModelCheckpoint(
            '/app/models/checkpoint_epoch_{epoch:02d}_batch_0000.h5',
            save_freq='epoch',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            patience=7,  # Reduced from 10 to 7 for better stability
            monitor='val_iou',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        CyclicalLearningRate(base_lr=5e-5, max_lr=5e-4, step_size=30, mode='triangular', verbose=1),
        MemoryCleanupCallback(),
        TrainingLoggerCallback('/app/models')
    ]
    
    # Train with gradient accumulation
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            workers=NUM_WORKERS,
            use_multiprocessing=False,
            max_queue_size=PREFETCH_SIZE,
            steps_per_epoch=max(1, len(train_gen) // GRADIENT_ACCUMULATION_STEPS),
            initial_epoch=initial_epoch - 1 if initial_epoch > 0 else 0
        )
        # Save final model in both formats
        print("\nSaving final models...")
        
        # Save training model (with dropout for potential retraining)
        model.save('/app/models/final_model_with_dropout.h5')
        print("Training model saved as 'final_model_with_dropout.h5'")
        
        # Create and save clean model for HLS4ML export (without dropout)
        print("Creating clean model for HLS4ML export...")
        clean_model = create_hls4ml_export_model(model)
        clean_model.save('/app/models/final_model_hls4ml_clean.h5')
        model_save_quantized_weights(clean_model, '/app/models/final_model_hls4ml_weights.h5')
        print("Clean HLS4ML model saved as 'final_model_hls4ml_clean.h5'")
        print("Quantized weights saved as 'final_model_hls4ml_weights.h5'")
        
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model weights only...")
        epoch = None
        batch = None
        if hasattr(model, 'stop_training') and hasattr(model, 'history'):
            epoch = model.history.epoch[-1] if hasattr(model.history, 'epoch') and model.history.epoch else None
            batch = 0
        if epoch is None:
            epoch = 0
        if batch is None:
            batch = 0
        checkpoint_name = f"/app/models/checkpoint_epoch_{epoch:02d}_batch_{batch:04d}_weights.h5"
        model.save_weights(checkpoint_name)
        print(f"Interrupted model weights saved as '{checkpoint_name}'")
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

#24/743 [..............................] - ETA: 3:05:34 - loss: 2.8576 - iou: 0.3420