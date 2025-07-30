import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path
import math

#  python3 /workspace/train.py

# SEMANTIC SEGMENTATION MODEL FOR HLS4ML (QAT-OPTIMIZED):
# 1. Memory-Efficient Data Loading:
#    - On-the-fly image loading with 10% random subset per epoch
#    - Dynamic augmentation during training
#    - No massive preprocessing or storage requirements
# 2. Quantization-Aware Training (QAT):
#    - QKeras layers (QConv2D, QBatchNormalization, QActivation)
#    - 8-bit quantization for FPGA compatibility
#    - L2 regularization and dropout for training stability
# 3. Standard Learning Rate Scheduling:
#    - ReduceLROnPlateau with gentle 30% reduction
#    - Monitors validation IoU for optimal performance
#    - Automatic early stopping to prevent overfitting
# 4. HLS4ML Compatibility:
#    - No unsupported layers (MaxPooling2D replaced with strided convolutions)
#    - Quantized weights export for FPGA synthesis
#    - Optimized for Xilinx VCU118 or similar high-end FPGAs

# Define constants
INPUT_HEIGHT = 512  # Reduced from 1024 to save memory
INPUT_WIDTH = 1024  # Reduced from 2048 to save memory
BATCH_SIZE = 4  # Reduced to 1 for memory constraints
EPOCHS = 100
NUM_CLASSES = 34
DATASET_DIR = '/data'  # Updated to match actual path
SUBSET_RATIO = 0.2  # Use 20% of dataset per epoch
NUM_WORKERS = 2  # Standard for data loading
PREFETCH_SIZE = 2  # Standard prefetch for efficiency
GRADIENT_ACCUMULATION_STEPS = 4  # No gradient accumulation needed with current setup

# Toggle for dropout usage (True for training, False for HLS4ML export)
USE_DROPOUT_DURING_TRAINING = False

# Toggle for HLS4ML compatible .h5 format on final save
SAVE_HLS4ML_COMPATIBLE = True



# Configure TensorFlow to use CPU efficiently
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)

# Disable eager execution for better memory tracking
# tf.compat.v1.disable_eager_execution()  # Commented out to enable eager execution

# Add a check to confirm eager execution is enabled
print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Add TF memory logger
tf.debugging.set_log_device_placement(False)  # Set to True only for debugging

# Disable layout optimizer to prevent conflicts with QKeras
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,  # Disabled to prevent QKeras conflicts
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})

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
    # Initial conv (use 16 bits) - reduced to 16
    x = conv_block(x, 16, name='init_conv', use_16bit=True)
    x = QConv2D(16, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample1')(x)
    # Stage 1 - reduced to 32
    x = c2f_block(x, 32, n=1, name='stage1')  # Reduced n from 2 to 1
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_stage1')(x)
    skip1 = x
    x = QConv2D(32, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample2')(x)
    # Stage 2 - reduced to 64
    x = c2f_block(x, 64, n=1, name='stage2')  # Reduced n from 2 to 1
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_stage2')(x)
    skip2 = x
    x = QConv2D(64, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample3')(x)
    # Stage 3 - reduced to 128
    x = c2f_block(x, 128, n=1, name='stage3')
    x = channel_attention(x, name='ca_stage3')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.3, name='dropout_stage3')(x)
    skip3 = x
    x = QConv2D(128, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_8),
                bias_quantizer=quantized_bits(**quantized_bits_param_8),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample4')(x)
    # Stage 4 - reduced to 256
    x = c2f_block(x, 256, n=1, name='stage4')
    x = channel_attention(x, name='ca_stage4')
    x = sppf_block(x, 256, name='sppf')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.35, name='dropout_stage4')(x)
    # Decoder (upsample and match filter counts)
    x = c2f_block(x, 256, n=1, name='pre_up1')
    # Use regular Conv2DTranspose instead of QConv2DTranspose for compatibility
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='up1_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat1')([x, skip3])
    x = conv_block(x, 128, name='up1_conv')
    x = c2f_block(x, 128, n=1, name='up1_c2f')
    x = channel_attention(x, name='ca_up1')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_up1')(x)
    x = c2f_block(x, 128, n=1, name='pre_up2')
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='up2_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat2')([x, skip2])
    x = conv_block(x, 64, name='up2_conv')
    x = c2f_block(x, 64, n=1, name='up2_c2f')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_up2')(x)
    x = c2f_block(x, 64, n=1, name='pre_up3')
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='up3_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat3')([x, skip1])
    x = conv_block(x, 32, name='up3_conv')
    x = c2f_block(x, 32, n=1, name='up3_c2f')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.15, name='dropout_up3')(x)
    x = c2f_block(x, 32, n=1, name='pre_up4')
    x = tf.keras.layers.Conv2DTranspose(16, kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='up4_transpose')(x)
    x = conv_block(x, 16, name='up4_conv')
    x = c2f_block(x, 16, n=1, name='up4_c2f')
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
                                print(f"Copied weights for layer: {clean_layer.name}")
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

class CityscapesDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, augment=False, preprocessed_prefix=None, subset_ratio=1.0):
        self.batch_size = batch_size
        self.augment = augment
        self.subset_ratio = subset_ratio  # Fraction of dataset to use per epoch (e.g., 0.1 for 10%)
        self.indices = None
        self.preprocessed = False
        self.images = None
        self.masks = None
        self.target_size = (INPUT_WIDTH, INPUT_HEIGHT)
        self.aug = None
        if augment:
            self.aug = A.Compose([
                # Basic geometric transformations (compatible with all versions)
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                
                # Basic color augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.2
                ),
                
                # Basic noise (updated for newer albumentations)
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                
                # Basic cutout (updated for newer albumentations)
                A.CoarseDropout(
                    max_holes=4,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    p=0.15
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
        
        # Select random subset for this epoch
        self._select_random_subset()

    def __len__(self):
        if self.preprocessed:
            return len(self.epoch_indices) // self.batch_size
        return len(self.epoch_indices) // self.batch_size

    def _select_random_subset(self):
        """Select a random subset of the dataset for this epoch"""
        if self.subset_ratio < 1.0:
            total_samples = len(self.indices)
            subset_size = int(total_samples * self.subset_ratio)
            # Randomly select subset_size samples
            self.epoch_indices = np.random.choice(self.indices, size=subset_size, replace=False)
            print(f"Selected {subset_size}/{total_samples} samples ({self.subset_ratio*100:.1f}%) for this epoch")
        else:
            self.epoch_indices = self.indices.copy()
    
    def on_epoch_end(self):
        # Select new random subset for next epoch
        self._select_random_subset()
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
        batch_indices = self.epoch_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
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

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        log_line = f"Epoch {epoch+1}: "
        log_line += ", ".join([f"{k}={v:.5f}" for k, v in logs.items() if isinstance(v, (float, int))])
        log_line += f" - lr: {lr:.6f}\n"
        log_line += f"Epoch {epoch+1}: saving model to /app/models/checkpoint_epoch_{epoch+1:02d}.h5\n"
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
        
        # Update state variables using direct assignment
        intersections_tensor = tf.stack(intersections)
        unions_tensor = tf.stack(unions)
        
        self.intersection.assign_add(intersections_tensor)
        self.union.assign_add(unions_tensor)

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



def train_model():
    # Configure GPU to use more VRAM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Allow TensorFlow to allocate more GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to use more VRAM (e.g., 90% of available)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024*10)]  # 8GB limit
                )
            print("✅ GPU memory growth enabled - VRAM will be allocated as needed")
            print("✅ GPU memory limit set to 8GB to utilize more VRAM")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")
    
    # Configure TensorFlow for optimal device placement BEFORE model creation
    # This will automatically place operations on the most suitable device
    tf.config.set_soft_device_placement(True)
    
    print("✅ Automatic CPU/GPU device placement enabled (hybrid mode)")
    print("✅ Device placement logging enabled - check output for CPU/GPU usage")
    
    # Create models directory if it doesn't exist
    os.makedirs('/app/models', exist_ok=True)
    
    # Find the latest saved model (check both .keras and .h5 files)
    model_files = [f for f in os.listdir('/app/models') if f.endswith(('.keras', '.h5'))]
    initial_epoch = 0
    skip_compilation = False  # Initialize compilation flag
    
    if model_files:
        # Sort by modification time to get the latest
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('/app/models', x)))
        print(f"\nFound latest model: {latest_model}")
        
        # Extract epoch number if it exists in filename
        if 'epoch_' in latest_model:
            try:
                # Handle both formats: checkpoint_epoch_15.keras/h5 and epoch_15.keras/h5
                if 'checkpoint_epoch_' in latest_model:
                    epoch_part = latest_model.split('checkpoint_epoch_')[1]
                else:
                    epoch_part = latest_model.split('epoch_')[1]
                
                # Remove .keras or .h5 extension and convert to int
                initial_epoch = int(epoch_part.split('.')[0])
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
            # Ensure device placement is respected during compilation
        )
        print("Model compiled successfully.")
    else:
        print("Skipping compilation - using loaded model with existing optimizer state.")
    
    # Setup data paths and prepare data
    cityscapes_root = '/data'
    print(f"\nLooking for Cityscapes dataset in: {cityscapes_root}")
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    # Validate data split to prevent data leakage (commented out for faster startup)
    # validate_data_split(train_images, train_masks, val_images, val_masks)
    
    print(f"\nFound {len(train_images)} training images and {len(val_images)} validation images")
    print(f"Using {SUBSET_RATIO*100:.0f}% random subset each epoch for faster training")
    
    # Create data generators with on-the-fly loading
    train_gen = CityscapesDataGenerator(train_images, train_masks, BATCH_SIZE, augment=True, subset_ratio=SUBSET_RATIO)
    val_gen = CityscapesDataGenerator(val_images, val_masks, BATCH_SIZE, augment=False, subset_ratio=SUBSET_RATIO)

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
            '/app/models/best_model.keras',
            save_best_only=True,
            monitor='val_iou',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        # Save model every epoch (standard approach)
        ModelCheckpoint(
            '/app/models/checkpoint_epoch_{epoch:02d}.keras',
            save_freq='epoch',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            patience=20,  # Increased from 7 to allow more training time
            monitor='val_iou',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_iou',
            mode='max',
            factor=0.7,  # Reduce LR by 30% when plateauing
            patience=3,   # Wait 3 epochs before reducing
            min_lr=1e-7,  # Minimum LR
            verbose=1,
            restore_best_weights=False
        ),
        MemoryCleanupCallback(),
        TrainingLoggerCallback('/app/models')
    ]
    
    # Train with automatic device placement for optimal performance
    try:
        print("Starting training with automatic CPU/GPU device placement...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            workers=NUM_WORKERS,
            use_multiprocessing=False,
            max_queue_size=PREFETCH_SIZE,
            steps_per_epoch=max(1, len(train_gen) // GRADIENT_ACCUMULATION_STEPS),
            initial_epoch= initial_epoch,
            verbose=1  # Force verbose output
            )
        print("Training completed successfully!")
        # Save final model based on dropout setting
        print("\nSaving final models...")
        
        if USE_DROPOUT_DURING_TRAINING:
            # Training with dropout - save training model only
            model.save('/app/models/final_model_with_dropout.keras')
            print("Training model saved as 'final_model_with_dropout.keras'")
            print("HLS4ML export skipped (dropout enabled - architectural mismatch)")
        else:
            # Training without dropout - save both training and HLS4ML models
            model.save('/app/models/final_model_no_dropout.keras')
            print("Training model saved as 'final_model_no_dropout.keras'")
            
            # Save HLS4ML compatible .h5 version if enabled
            if SAVE_HLS4ML_COMPATIBLE:
                model.save('/app/models/final_model_hls4ml.h5', save_format='h5')
                print("HLS4ML compatible model saved as 'final_model_hls4ml.h5'")
            
            # Create and save clean model for HLS4ML export (identical architecture)
            print("Creating HLS4ML export model...")
            clean_model = create_hls4ml_export_model(model)
            clean_model.save('/app/models/final_model_hls4ml_clean.keras')
            model_save_quantized_weights(clean_model, '/app/models/final_model_hls4ml_weights.h5')
            print("Clean HLS4ML model saved as 'final_model_hls4ml_clean.keras'")
            print("Quantized weights saved as 'final_model_hls4ml_weights.h5'")
        
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current full model...")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"/app/models/interrupted_model_{timestamp}.h5"
        model.save(model_path)
        print(f"Interrupted full model saved as '{model_path}'")
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

#24/743 [..............................] - ETA: 3:05:34 - loss: 2.8576 - iou: 0.3420

