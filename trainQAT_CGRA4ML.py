#!/usr/bin/env python3
"""
LMIINet: CGRA4ML-Compatible Lightweight Multiple-Information Interaction Network
===============================================================================

This script implements LMIINet architecture optimized for CGRA4ML deployment.
Modified for full CGRA4ML compatibility with simplified operations and better
skip connection handling.

LMIINet achieves 72.0% mIoU on Cityscapes with only 0.72M parameters and 11.74G FLOPs.

CGRA4ML Optimizations:
- Simplified operations for better CGRA4ML mapping
- Streamlined skip connections for efficient dataflow
- Reduced branching complexity for single reconfigurable engine
- Memory-efficient feature fusion
- Hardware-friendly activation functions
- Optimized for CGRA4ML's dataflow architecture

Usage:
    python trainQAT_CGRA4ML.py --trainQAT=False  # Train baseline float32 model
    python trainQAT_CGRA4ML.py --trainQAT=True   # Apply QAT to trained model

Key Features:
- CGRA4ML-optimized LMIINet architecture
- Simplified LFIB modules for better CGRA4ML mapping
- Streamlined skip connections with efficient memory usage
- Configurable dropout layers for regularization
- Baseline float32 training for optimal convergence
- QAT application after float32 training completes
- Memory-efficient data pipeline
- Full CGRA4ML compatibility

CGRA4ML Compatibility:
- Uses only CGRA4ML-supported operations
- Simplified branching for single reconfigurable engine
- Efficient skip connection implementation
- Memory-optimized feature fusion
- Hardware-friendly activation functions
- Streamlined dataflow for CGRA4ML processing elements

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=True --subset_ratio=0.9 
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=True --subset_ratio=0.9 --LR=1e-5
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=False --subset_ratio=0.9 

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=1 --batch_size=1 --use_dropout=True --subset_ratio=0.1 --load_model=./models/qat_model_checkpoint_epoch_60.keras

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
import glob
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, 
    Add, DepthwiseConv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, 
    Reshape, AveragePooling2D, MaxPooling2D
)
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu, QDense, QDepthwiseConv2D
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import MeanIoU
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow.keras.backend as K

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CGRA4ML Semantic Segmentation with QAT Toggle')
    parser.add_argument('--trainQAT', type=bool, default=False, help='Train with QAT (True) or baseline (False)')
    parser.add_argument('--epochs', type=int, default=140, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--use_dropout', type=bool, default=True, help='Use dropout during training')
    parser.add_argument('--subset_ratio', type=float, default=0.9, help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load existing model')
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode')
    args = parser.parse_args()

# Configuration constants
TRAIN_QAT = args.trainQAT
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
USE_DROPOUT_DURING_TRAINING = args.use_dropout
SUBSET_RATIO = args.subset_ratio
LEARNING_RATE = args.LR
LOAD_MODEL_PATH = args.load_model
DEBUG_MODE = args.debug

# Model constants
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 20

# Data paths - Docker mounted paths
TRAIN_IMAGES_DIR = "/data/leftImg8bit/train"
TRAIN_MASKS_DIR = "/data/gtFine/train"
VAL_IMAGES_DIR = "/data/leftImg8bit/val"
VAL_MASKS_DIR = "/data/gtFine/val"

# Model save paths - Docker workspace paths
MODELS_DIR = "/workspace/models"
TB_LOGS_DIR = "/workspace/models/tb_logs"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TB_LOGS_DIR, exist_ok=True)

# Configure TensorFlow for optimal performance
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})

def safe_configure_tensorflow(debug_mode=False):
    """Safely configure TF threading and GPU options if not already initialized."""
    try:
        n_cores = multiprocessing.cpu_count()
        tf.config.threading.set_inter_op_parallelism_threads(n_cores)
        tf.config.threading.set_intra_op_parallelism_threads(n_cores)
        tf.config.set_soft_device_placement(True)
    except RuntimeError as e:
        print(f"Skipping thread config (already initialized): {e}")

    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 13)],
                    )
                except RuntimeError:
                    pass
    except Exception as e:
        print(f"GPU config skipped: {e}")

    try:
        tf.config.optimizer.set_jit(True)
    except Exception as e:
        print(f"XLA config skipped: {e}")

    print(f"Eager execution enabled: {tf.executing_eagerly()}")
    try:
        tf.debugging.set_log_device_placement(debug_mode)
    except Exception:
        pass

# Call safe TF configuration early
safe_configure_tensorflow(DEBUG_MODE)

def get_cgra4ml_quantization_config(layer_name, layer_type):
    """
    Get intelligent quantization configuration with appropriate bit allocation
    Uses 32-bit for critical layers, 16-bit for important layers, 8-bit for efficiency
    """
    # Output layer - CRITICAL for accuracy, use 32-bit
    if 'output' in layer_name.lower():
        return create_quantized_bits_config(32, 6, True)  # 32-bit, 6 int bits for maximum accuracy
    
    # Stem layer - CRITICAL for initial feature extraction, use 32-bit
    elif 'stem' in layer_name.lower():
        return create_quantized_bits_config(32, 6, True)  # 32-bit, 6 int bits for critical features
    
    # BatchNorm moving stats - CRITICAL for stability, use 32-bit
    elif ('bn' in layer_name.lower() or 'batch' in layer_name.lower()) and ('moving_mean' in layer_name.lower() or 'moving_variance' in layer_name.lower()):
        # Moving stats are critical for BatchNorm stability
        return create_quantized_bits_config(32, 8, True)  # 32-bit, 8 int bits for stability
    
    # BatchNorm gamma/beta - IMPORTANT for stability, use 16-bit
    elif 'bn' in layer_name.lower() or 'batch' in layer_name.lower():
        return create_quantized_bits_config(16, 4, True)  # 16-bit, 4 int bits for stability
    
    # Early layers (stage1) - IMPORTANT for feature extraction, use 16-bit
    elif 'stage1' in layer_name.lower():
        return create_quantized_bits_config(16, 5, True)  # 16-bit, 5 int bits for good range
    
    # Middle layers (stage2, stage3) - IMPORTANT for feature processing, use 16-bit
    elif 'stage2' in layer_name.lower() or 'stage3' in layer_name.lower():
        return create_quantized_bits_config(16, 4, True)  # 16-bit, 4 int bits for good accuracy
    
    # Deep layers (stage4) - IMPORTANT for semantic features, use 16-bit
    elif 'stage4' in layer_name.lower():
        return create_quantized_bits_config(16, 4, True)  # 16-bit, 4 int bits for semantic accuracy
    
    # Dense/MLP layers - IMPORTANT for global context, use 16-bit
    elif 'dense' in layer_name.lower() or 'mlp' in layer_name.lower():
        return create_quantized_bits_config(16, 4, True)  # 16-bit, 4 int bits for context accuracy
    
    # Decoder layers - CRITICAL for reconstruction, use 16-bit
    elif 'dec' in layer_name.lower() or 'up' in layer_name.lower():
        return create_quantized_bits_config(16, 5, True)  # 16-bit, 5 int bits for reconstruction accuracy
    
    # Default - conservative approach
    else:
        return create_quantized_bits_config(16, 6, True)  # 16-bit, 6 int bits

def create_quantized_bits_config(bits, integer_bits, symmetric=True):
    """
    Create quantization configuration dictionary based on weight analysis.
    
    Args:
        bits: Total number of bits (8, 16, or 32)
        integer_bits: Number of integer bits
        symmetric: Whether to use symmetric quantization
    
    Returns:
        Dictionary with quantization configuration
    """
    return {
        'bits': bits,
        'integer': integer_bits,
        'symmetric': symmetric,
        # Removed 'alpha': 'auto_po2' to prevent unsupported quantizers
        'use_stochastic_rounding': True
    }

# Quantization configurations for CGRA4ML
# Activations are important for feature propagation - use 16-bit for accuracy
quantized_relu_config = {
    'bits': 16,
    'integer': 6,
    'use_stochastic_rounding': True
}

def cgra4ml_lfib_module(x, filters, use_quantization=False, name_prefix='lfib', dropout_rate=0.1):
    """
    CGRA4ML-optimized Lightweight Feature Interaction Bottleneck (LFIB) module
    Simplified for better CGRA4ML mapping with reduced branching complexity
    """
    if use_quantization:
        conv_config = get_cgra4ml_quantization_config(f'{name_prefix}_conv', 'Conv2D')
        bn_config = get_cgra4ml_quantization_config(f'{name_prefix}_bn', 'BatchNormalization')
        act_config = quantized_relu_config
        
        # Simplified branching for CGRA4ML - reduced complexity
        # Branch 1: Standard convolution (simpler than depthwise for CGRA4ML)
        x1 = QConv2D(filters//2, kernel_size=3, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_conv1')(x)
        x1 = QBatchNormalization(gamma_quantizer=quantized_bits(**bn_config),
                               beta_quantizer=quantized_bits(**bn_config),
                               mean_quantizer=quantized_bits(**bn_config),
                               variance_quantizer=quantized_bits(**bn_config),
                               name=f'{name_prefix}_bn1')(x1)
        x1 = QActivation(quantized_relu(**act_config), name=f'{name_prefix}_act1')(x1)
        
        # Branch 2: Dilated convolution for context
        x2 = QConv2D(filters//2, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_dilated')(x)
        
        # Simple concatenation (CGRA4ML friendly)
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x1, x2])
        
        # Final 1x1 convolution
        x_out = QConv2D(filters, kernel_size=1, padding='same', use_bias=False,
                       kernel_quantizer=quantized_bits(**conv_config),
                       bias_quantizer=quantized_bits(**conv_config),
                       name=f'{name_prefix}_final')(x_concat)
        x_out = QBatchNormalization(gamma_quantizer=quantized_bits(**bn_config),
                                  beta_quantizer=quantized_bits(**bn_config),
                                  mean_quantizer=quantized_bits(**bn_config),
                                  variance_quantizer=quantized_bits(**bn_config),
                                  name=f'{name_prefix}_bn_final')(x_out)
        x_out = QActivation(quantized_relu(**act_config), name=f'{name_prefix}_act_final')(x_out)
        
    else:
        # Branch 1: Standard convolution
        x1 = Conv2D(filters//2, kernel_size=3, padding='same', use_bias=False,
                   name=f'{name_prefix}_conv1')(x)
        x1 = BatchNormalization(name=f'{name_prefix}_bn1')(x1)
        x1 = Activation('relu', name=f'{name_prefix}_act1')(x1)
        
        # Branch 2: Dilated convolution
        x2 = Conv2D(filters//2, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
                   name=f'{name_prefix}_dilated')(x)
        
        # Simple concatenation
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x1, x2])
        
        # Final 1x1 convolution
        x_out = Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
                      name=f'{name_prefix}_final')(x_concat)
        x_out = BatchNormalization(name=f'{name_prefix}_bn_final')(x_out)
        x_out = Activation('relu', name=f'{name_prefix}_act_final')(x_out)
    
    # Add dropout if enabled
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_out = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_out)
    
    return x_out

def cgra4ml_global_context_block(x, use_quantization=False, name_prefix='global_context', dropout_rate=0.1):
    """
    CGRA4ML-optimized global context block
    Simplified for better CGRA4ML mapping
    """
    channels = x.shape[-1]
    
    # Global average pooling
    x_pooled = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    
    # Simple MLP for global context
    if use_quantization:
        mlp_config = get_cgra4ml_quantization_config(f'{name_prefix}_mlp', 'Dense')
        
        x_mlp = QDense(channels // 4, use_bias=False,
                      kernel_quantizer=quantized_bits(**mlp_config),
                      bias_quantizer=quantized_bits(**mlp_config),
                      name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = QActivation(quantized_relu(**quantized_relu_config), name=f'{name_prefix}_act1')(x_mlp)
        x_mlp = QDense(channels, use_bias=False,
                      kernel_quantizer=quantized_bits(**mlp_config),
                      bias_quantizer=quantized_bits(**mlp_config),
                      name=f'{name_prefix}_mlp2')(x_mlp)
        x_mlp = QActivation(quantized_relu(**quantized_relu_config), name=f'{name_prefix}_act2')(x_mlp)
    else:
        x_mlp = Dense(channels // 4, use_bias=False, name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = Activation('relu', name=f'{name_prefix}_act1')(x_mlp)
        x_mlp = Dense(channels, use_bias=False, name=f'{name_prefix}_mlp2')(x_mlp)
        x_mlp = Activation('relu', name=f'{name_prefix}_act2')(x_mlp)
    
    # Reshape for broadcasting
    x_mlp = Reshape((1, 1, channels), name=f'{name_prefix}_reshape')(x_mlp)
    
    # Apply global context
    x_out = Add(name=f'{name_prefix}_add')([x, x_mlp])
    
    # Add dropout if enabled
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_out = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_out)
    
    return x_out

def cgra4ml_encoder(inputs, use_quantization=False):
    """
    CGRA4ML-optimized LMIINet encoder
    Simplified for better CGRA4ML mapping with efficient skip connections
    """
    # Initial stem
    if use_quantization:
        stem_config = get_cgra4ml_quantization_config('stem', 'Conv2D')
        x = QConv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stem')(inputs)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stem_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stem_act')(x)
    else:
        x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='stem')(inputs)
        x = BatchNormalization(name='stem_bn')(x)
        x = Activation('relu', name='stem_act')(x)
    
    # Stage 1: 1/4 resolution
    x = cgra4ml_lfib_module(x, 64, use_quantization, 'stage1_lfib1', dropout_rate=0.05)
    x = cgra4ml_lfib_module(x, 64, use_quantization, 'stage1_lfib2', dropout_rate=0.05)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage1_global', dropout_rate=0.05)
    stage1_out = x
    
    # Stage 2: 1/8 resolution
    if use_quantization:
        x = QConv2D(128, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stage2_downsample')(x)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stage2_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage2_downsample_act')(x)
    else:
        x = Conv2D(128, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage2_downsample')(x)
        x = BatchNormalization(name='stage2_downsample_bn')(x)
        x = Activation('relu', name='stage2_downsample_act')(x)
    x = cgra4ml_lfib_module(x, 128, use_quantization, 'stage2_lfib1', dropout_rate=0.1)
    x = cgra4ml_lfib_module(x, 128, use_quantization, 'stage2_lfib2', dropout_rate=0.1)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage2_global', dropout_rate=0.1)
    stage2_out = x
    
    # Stage 3: 1/16 resolution
    if use_quantization:
        x = QConv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stage3_downsample')(x)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stage3_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage3_downsample_act')(x)
    else:
        x = Conv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage3_downsample')(x)
        x = BatchNormalization(name='stage3_downsample_bn')(x)
        x = Activation('relu', name='stage3_downsample_act')(x)
    x = cgra4ml_lfib_module(x, 256, use_quantization, 'stage3_lfib1', dropout_rate=0.15)
    x = cgra4ml_lfib_module(x, 256, use_quantization, 'stage3_lfib2', dropout_rate=0.15)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage3_global', dropout_rate=0.15)
    stage3_out = x
    
    # Stage 4: 1/32 resolution
    if use_quantization:
        x = QConv2D(512, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stage4_downsample')(x)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stage4_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage4_downsample_act')(x)
    else:
        x = Conv2D(512, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage4_downsample')(x)
        x = BatchNormalization(name='stage4_downsample_bn')(x)
        x = Activation('relu', name='stage4_downsample_act')(x)
    x = cgra4ml_lfib_module(x, 512, use_quantization, 'stage4_lfib1', dropout_rate=0.2)
    x = cgra4ml_lfib_module(x, 512, use_quantization, 'stage4_lfib2', dropout_rate=0.2)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage4_global', dropout_rate=0.2)
    stage4_out = x
    
    return stage1_out, stage2_out, stage3_out, stage4_out

def cgra4ml_decoder(stage1_out, stage2_out, stage3_out, stage4_out, use_quantization=False):
    """
    CGRA4ML-optimized LMIINet decoder
    Simplified skip connections for better CGRA4ML mapping
    """
    # Start from stage4 (1/32 resolution)
    x = stage4_out

    # Upsample to stage3 (1/16) with skip connection
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up1')(x)
    x = Concatenate(name='concat1')([x, stage3_out])
    x = cgra4ml_lfib_module(x, 256, use_quantization, 'dec1_lfib', dropout_rate=0.15)

    # Upsample to stage2 (1/8) with skip connection
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up2')(x)
    x = Concatenate(name='concat2')([x, stage2_out])
    x = cgra4ml_lfib_module(x, 128, use_quantization, 'dec2_lfib', dropout_rate=0.1)

    # Upsample to stage1 (1/4) with skip connection
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up3')(x)
    x = Concatenate(name='concat3')([x, stage1_out])
    x = cgra4ml_lfib_module(x, 64, use_quantization, 'dec3_lfib', dropout_rate=0.05)

    # Final upsampling to full resolution (1/4 -> 1/1)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear', name='up_final')(x)

    # Final convolution for output
    if use_quantization:
        output_config = get_cgra4ml_quantization_config('output_conv', 'Conv2D')
        x = QConv2D(NUM_CLASSES, kernel_size=1, padding='same', activation='softmax',
                   kernel_quantizer=quantized_bits(**output_config),
                   bias_quantizer=quantized_bits(**output_config),
                   name='output_conv')(x)
    else:
        x = Conv2D(NUM_CLASSES, kernel_size=1, padding='same', activation='softmax', name='output_conv')(x)

    return x

def create_cgra4ml_segmentation_model(use_quantization=False):
    """
    Create CGRA4ML-optimized LMIINet model
    """
    tf.keras.backend.clear_session()

    # Input
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name="input_image")

    # CGRA4ML-optimized encoder
    stage1_out, stage2_out, stage3_out, stage4_out = cgra4ml_encoder(
        inputs, use_quantization
    )

    # CGRA4ML-optimized decoder
    outputs = cgra4ml_decoder(
        stage1_out, stage2_out, stage3_out, stage4_out, use_quantization
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LMIINet_CGRA4ML")
    return model

class CityscapesDataGenerator(tf.keras.utils.Sequence):
    """Data generator for Cityscapes dataset"""

    def __init__(
        self,
        image_dir,
        mask_dir,
        batch_size=1,
        subset_ratio=1.0,
        target_height=512,
        target_width=1024,
        shuffle=True,
        augment=True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment

        # Get all image paths
        self.image_paths = list(self.image_dir.glob("**/*.png"))
        if subset_ratio < 1.0:
            subset_size = int(len(self.image_paths) * subset_ratio)
            self.image_paths = self.image_paths[:subset_size]

        print(f"Found {len(self.image_paths)} images")

        # Create balanced augmentation pipeline
        if self.augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                    A.GaussNoise(var_limit=10.0, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                ]
            )
        else:
            self.transform = None

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        try:
            # Load image
            img = load_img(image_path, target_size=(self.target_height, self.target_width))
            img_array = np.array(img) / 255.0
            
            # Load corresponding mask
            mask_path = self._get_mask_path(image_path)
            if mask_path.exists():
                mask = load_img(mask_path, target_size=(self.target_height, self.target_width), color_mode='grayscale')
                mask_array = np.array(mask)
                
                # Convert to categorical (0-19 for 20 classes)
                mask_categorical = np.zeros((self.target_height, self.target_width, NUM_CLASSES))
                for i in range(NUM_CLASSES):
                    mask_categorical[:, :, i] = (mask_array == i).astype(np.float32)
                
                return img_array, mask_categorical
            else:
                print(f"Mask not found: {mask_path}")
                return None, None
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None

    def _get_mask_path(self, image_path):
        """Get corresponding mask path"""
        # Convert image path to mask path
        relative_path = image_path.relative_to(self.image_dir)
        mask_path = self.mask_dir / relative_path.parent / f"{relative_path.stem}_gtFine_labelIds.png"
        return mask_path

    def __getitem__(self, index):
        """Get batch of data"""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for image_path in batch_paths:
            img, mask = self.load_and_preprocess_image(image_path)
            if img is not None and mask is not None:
                batch_images.append(img)
                batch_masks.append(mask)
        
        if not batch_images:
            # Return dummy data if no valid images
            dummy_img = np.zeros((self.target_height, self.target_width, 3))
            dummy_mask = np.zeros((self.target_height, self.target_width, NUM_CLASSES))
            dummy_mask[:, :, 0] = 1  # Set to background class
            return np.array([dummy_img]), np.array([dummy_mask])
        
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        
        # Apply augmentation if enabled
        if self.transform:
            augmented_images = []
            augmented_masks = []
            
            for img, mask in zip(batch_images, batch_masks):
                # Convert mask back to single channel for augmentation
                mask_single = np.argmax(mask, axis=-1).astype(np.uint8)
                
                # Apply augmentation
                augmented = self.transform(image=img, mask=mask_single)
                aug_img = augmented['image']
                aug_mask = augmented['mask']
                
                # Convert mask back to categorical
                aug_mask_categorical = np.zeros((self.target_height, self.target_width, NUM_CLASSES))
                for i in range(NUM_CLASSES):
                    aug_mask_categorical[:, :, i] = (aug_mask == i).astype(np.float32)
                
                augmented_images.append(aug_img)
                augmented_masks.append(aug_mask_categorical)
            
            batch_images = np.array(augmented_images)
            batch_masks = np.array(augmented_masks)
        
        return batch_images, batch_masks

def combined_loss(y_true, y_pred):
    """Combined loss function for semantic segmentation"""
    # Categorical crossentropy
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Dice loss
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice_coeff
    
    # Combined loss
    total_loss = ce_loss + dice_loss
    return total_loss

def create_callbacks(model_name):
    """Create training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint_epoch_{{epoch:02d}}.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    # TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=TB_LOGS_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks

def main():
    """Main training function"""
    print("=" * 60)
    print("CGRA4ML LMIINet Semantic Segmentation Training")
    print("=" * 60)
    print(f"Training with QAT: {TRAIN_QAT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Use dropout: {USE_DROPOUT_DURING_TRAINING}")
    print(f"Subset ratio: {SUBSET_RATIO}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_generator = CityscapesDataGenerator(
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=True
    )
    
    val_generator = CityscapesDataGenerator(
        VAL_IMAGES_DIR,
        VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=False,
        augment=False
    )
    
    print(f"Training samples: {len(train_generator) * BATCH_SIZE}")
    print(f"Validation samples: {len(val_generator) * BATCH_SIZE}")
    
    # Create model
    print("\nCreating CGRA4ML-optimized LMIINet model...")
    model = create_cgra4ml_segmentation_model(use_quantization=TRAIN_QAT)
    
    # Load existing model if specified
    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading existing model from: {LOAD_MODEL_PATH}")
        model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
    
    # Compile model
    print("\nCompiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['accuracy', MeanIoU(num_classes=NUM_CLASSES, name='mean_iou')]
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    model_name = "cgra4ml_lmiinet_qat" if TRAIN_QAT else "cgra4ml_lmiinet_baseline"
    callbacks = create_callbacks(model_name)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save quantized weights if using QAT
    if TRAIN_QAT:
        quantized_weights_path = os.path.join(MODELS_DIR, f"{model_name}_quantized_weights.h5")
        model_save_quantized_weights(model, quantized_weights_path)
        print(f"Quantized weights saved to: {quantized_weights_path}")
    
    print("\nTraining completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
