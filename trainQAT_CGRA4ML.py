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

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=1 --batch_size=1 --use_dropout=True --subset_ratio=0.1
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
import traceback
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
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

# Model configuration for CGRA4ML-optimized LMIINet
# This configuration is used throughout the model architecture to ensure consistency
# and match the original LMIINet parameter count (~2.7M parameters)
MODEL_CONFIG = {
    # MobileNetV2 channels (fixed to match DeepLabV3+ MobileNetV2 ~2.7M params)
    'mv2_init': 16,        # Initial stem channels
    'mv2_c1':   16,        # Stage 1 channels (1/4 resolution)
    'mv2_c2':   24,        # Stage 2 channels (1/8 resolution)
    'mv2_c3':   32,        # Stage 3 channels (1/16 resolution)
    'mv2_c4':   64,        # Stage 4 channels (1/32 resolution)
    'mv2_c5':   96,        # Additional stage channels (if needed)
    'mv2_c6':   160,       # Additional stage channels (if needed)
    'mv2_c7':   320,       # Additional stage channels (if needed)
    # ASPP and decoder configuration
    'aspp_filters':    128, # Global context MLP filters
    'low_level_proj':  32,  # Low-level feature projection
    'decoder_filters': 128, # Decoder refinement filters
    # Use dilated convolutions in decoder refinement blocks
    'use_dilated_conv': True,
}

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

    # Enable eager execution for QKeras compatibility
    # tf.config.run_functions_eagerly(True)
    
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
        'use_stochastic_rounding': False  # Disable to avoid learning_phase() issue
    }

# Quantization configurations for CGRA4ML
# Activations are important for feature propagation - use 16-bit for accuracy
quantized_relu_config = {
    'bits': 16,
    'integer': 6,
    'use_stochastic_rounding': False  # Disable to avoid learning_phase() issue
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
        x1 = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
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
        x_out = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
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
        
        x_mlp = QDense(MODEL_CONFIG['aspp_filters'], use_bias=False,
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
        x_mlp = Dense(MODEL_CONFIG['aspp_filters'], use_bias=False, name=f'{name_prefix}_mlp1')(x_pooled)
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
        bn_config = get_cgra4ml_quantization_config('stem_bn', 'BatchNormalization')
        x = QConv2D(MODEL_CONFIG['mv2_init'], kernel_size=3, strides=2, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**stem_config),
                    bias_quantizer=quantized_bits(**stem_config),
                    name='stem')(inputs)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
            beta_quantizer=quantized_bits(**bn_config),
            mean_quantizer=quantized_bits(**bn_config),
            variance_quantizer=quantized_bits(**bn_config),
            name='stem_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stem_act')(x)
    else:
        x = Conv2D(MODEL_CONFIG['mv2_init'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stem')(inputs)
        x = BatchNormalization(name='stem_bn')(x)
        x = Activation('relu', name='stem_act')(x)
    
    # Stage 1: 1/4 resolution (additional stride 2)
    if use_quantization:
        bn_config = get_cgra4ml_quantization_config('stage1_downsample_bn', 'BatchNormalization')
        x = QConv2D(MODEL_CONFIG['mv2_c1'], kernel_size=3, strides=2, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**stem_config),
                    bias_quantizer=quantized_bits(**stem_config),
                    name='stage1_downsample')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
            beta_quantizer=quantized_bits(**bn_config),
            mean_quantizer=quantized_bits(**bn_config),
            variance_quantizer=quantized_bits(**bn_config),
            name='stage1_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage1_downsample_act')(x)
    else:
        x = Conv2D(MODEL_CONFIG['mv2_c1'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage1_downsample')(x)
        x = BatchNormalization(name='stage1_downsample_bn')(x)
        x = Activation('relu', name='stage1_downsample_act')(x)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c1'], use_quantization, 'stage1_lfib1', dropout_rate=0.05)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c1'], use_quantization, 'stage1_lfib2', dropout_rate=0.05)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage1_global', dropout_rate=0.05)
    stage1_out = x
    
    # Stage 2: 1/8 resolution
    if use_quantization:
        bn_config = get_cgra4ml_quantization_config('stage2_downsample_bn', 'BatchNormalization')
        x = QConv2D(MODEL_CONFIG['mv2_c2'], kernel_size=3, strides=2, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**stem_config),
                    bias_quantizer=quantized_bits(**stem_config),
                    name='stage2_downsample')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
            beta_quantizer=quantized_bits(**bn_config),
            mean_quantizer=quantized_bits(**bn_config),
            variance_quantizer=quantized_bits(**bn_config),
            name='stage2_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage2_downsample_act')(x)
    else:
        x = Conv2D(MODEL_CONFIG['mv2_c2'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage2_downsample')(x)
        x = BatchNormalization(name='stage2_downsample_bn')(x)
        x = Activation('relu', name='stage2_downsample_act')(x)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c2'], use_quantization, 'stage2_lfib1', dropout_rate=0.1)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c2'], use_quantization, 'stage2_lfib2', dropout_rate=0.1)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage2_global', dropout_rate=0.1)
    stage2_out = x
    
    # Stage 3: 1/16 resolution
    if use_quantization:
        bn_config = get_cgra4ml_quantization_config('stage3_downsample_bn', 'BatchNormalization')
        x = QConv2D(MODEL_CONFIG['mv2_c3'], kernel_size=3, strides=2, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**stem_config),
                    bias_quantizer=quantized_bits(**stem_config),
                    name='stage3_downsample')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
            beta_quantizer=quantized_bits(**bn_config),
            mean_quantizer=quantized_bits(**bn_config),
            variance_quantizer=quantized_bits(**bn_config),
            name='stage3_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage3_downsample_act')(x)
    else:
        x = Conv2D(MODEL_CONFIG['mv2_c3'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage3_downsample')(x)
        x = BatchNormalization(name='stage3_downsample_bn')(x)
        x = Activation('relu', name='stage3_downsample_act')(x)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c3'], use_quantization, 'stage3_lfib1', dropout_rate=0.15)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c3'], use_quantization, 'stage3_lfib2', dropout_rate=0.15)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage3_global', dropout_rate=0.15)
    stage3_out = x
    
    # Stage 4: 1/32 resolution
    if use_quantization:
        bn_config = get_cgra4ml_quantization_config('stage4_downsample_bn', 'BatchNormalization')
        x = QConv2D(MODEL_CONFIG['mv2_c4'], kernel_size=3, strides=2, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**stem_config),
                    bias_quantizer=quantized_bits(**stem_config),
                    name='stage4_downsample')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**bn_config),
            beta_quantizer=quantized_bits(**bn_config),
            mean_quantizer=quantized_bits(**bn_config),
            variance_quantizer=quantized_bits(**bn_config),
            name='stage4_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage4_downsample_act')(x)
    else:
        x = Conv2D(MODEL_CONFIG['mv2_c4'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage4_downsample')(x)
        x = BatchNormalization(name='stage4_downsample_bn')(x)
        x = Activation('relu', name='stage4_downsample_act')(x)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c4'], use_quantization, 'stage4_lfib1', dropout_rate=0.2)
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['mv2_c4'], use_quantization, 'stage4_lfib2', dropout_rate=0.2)
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
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['decoder_filters'], use_quantization, 'dec1_lfib', dropout_rate=0.15)

    # Upsample to stage2 (1/8) with skip connection
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up2')(x)
    x = Concatenate(name='concat2')([x, stage2_out])
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['decoder_filters'], use_quantization, 'dec2_lfib', dropout_rate=0.1)

    # Upsample to stage1 (1/4) with skip connection
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up3')(x)
    x = Concatenate(name='concat3')([x, stage1_out])
    x = cgra4ml_lfib_module(x, MODEL_CONFIG['low_level_proj'], use_quantization, 'dec3_lfib', dropout_rate=0.05)

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
        **kwargs
    ):
        super().__init__(**kwargs)  # Call parent constructor to fix PyDataset warning
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

        # Create balanced augmentation pipeline with reduced probabilities
        if self.augment:
            self.transform = A.Compose(
                [
                    # Geometric augmentations - more aggressive for robustness
                    A.RandomCrop(
                        width=self.target_width, height=self.target_height, p=0.1
                    ),
                    A.HorizontalFlip(p=0.2),
                    A.Rotate(limit=5, p=0.2, border_mode=cv2.BORDER_CONSTANT),
                    A.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.1
                    ),
                    # Color and contrast augmentations - enhanced for lighting variations
                    A.RandomBrightnessContrast(
                        brightness_limit=0.4, contrast_limit=0.4, p=0.3
                    ),
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=30,
                        p=0.3,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                    # Noise and blur augmentations
                    A.GaussNoise(var_limit=10.0, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                    A.MotionBlur(blur_limit=3, p=0.15),
                    # Weather and lighting effects
                    A.RandomRain(
                        drop_length=20,
                        drop_width=1,
                        drop_color=(200, 200, 200),
                        blur_value=3,
                        brightness_coefficient=0.7,
                        rain_type="drizzle",
                        p=0.1,
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.1
                    ),
                    # Occlusion and masking
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=16,
                        max_width=16,
                        min_holes=1,
                        min_height=4,
                        min_width=4,
                        p=0.15,
                    ),
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
        """Load and preprocess a single image with robust Cityscapes handling"""
        try:
            # Load image using cv2 for better control
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load corresponding mask with robust Cityscapes handling
            mask_path = self._get_mask_path(image_path)
            if not mask_path.exists():
                print(f"Mask not found: {mask_path}")
                return None, None

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")

            # Handle different mask formats
            if "_labelTrainIds.png" in str(mask_path):
                # Already in correct format (0..18 with 255 ignore)
                pass
            else:
                # Map labelIds (34) to our 20-class scheme: 19 trainIds + void=19
                lid_to_tid = {
                    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 18: 5, 19: 6, 20: 7,
                    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
                    31: 16, 32: 17, 33: 18, 29: 13, 30: 13,
                }
                mapped = np.full_like(mask, 19, dtype=np.uint8)
                for lid, tid in lid_to_tid.items():
                    mapped[mask == lid] = tid
                mask = mapped

            # Resize image and mask
            image = cv2.resize(
                image,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Normalize image
            image = image.astype(np.float32) / 255.0

            # One-hot: map 255 (ignore) to void index 19
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask_safe = mask.copy()
            mask_safe[mask_safe == 255] = 19
            mask = tf.keras.utils.to_categorical(mask_safe, num_classes=NUM_CLASSES)

            return image, mask

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            dummy_image = np.zeros(
                (self.target_height, self.target_width, 3), dtype=np.float32
            )
            dummy_mask = np.zeros(
                (self.target_height, self.target_width, NUM_CLASSES), dtype=np.float32
            )
            dummy_mask[:, :, 0] = 1  # Set background class
            return dummy_image, dummy_mask

    def _get_mask_path(self, image_path):
        """Get corresponding mask path with robust Cityscapes handling"""
        # 1) Prefer *_labelTrainIds.png (already 0..18 with 255 ignore)
        mask_name_tids = image_path.name.replace(
            "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
        )
        mask_path_tids = self.mask_dir / image_path.parent.name / mask_name_tids

        if mask_path_tids.exists():
            return mask_path_tids
        else:
            # 2) Fallback to *_labelIds.png (34-class)
            mask_name_lids = image_path.name.replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"
            )
            mask_path_lids = self.mask_dir / image_path.parent.name / mask_name_lids
            return mask_path_lids

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_paths = self.image_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []

        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            if image is not None and mask is not None:
                batch_images.append(image)
                batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for semantic segmentation - measures overlap between predicted and ground truth masks
    """
    # Ensure consistent tensor types for mixed precision compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute per-class Dice over classes 0..18 (exclude void=19), unweighted
    y_true_19 = y_true[..., :19]
    y_pred_19 = y_pred[..., :19]
    y_true_clipped = K.clip(y_true_19, 0.0, 1.0)
    y_pred_clipped = K.clip(y_pred_19, 0.0, 1.0)
    intersection = K.sum(y_true_clipped * y_pred_clipped, axis=[0, 1, 2])
    sums = K.sum(y_true_clipped, axis=[0, 1, 2]) + K.sum(y_pred_clipped, axis=[0, 1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (sums + smooth)
    return K.mean(1.0 - dice_per_class)

def mean_iou(y_true, y_pred):
    """
    Mean IoU metric for semantic segmentation - ignores void class (index 19)
    """
    # Convert to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    # Calculate IoU for each class
    ious = []
    for class_id in range(NUM_CLASSES - 1):  # exclude void=19
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true == class_id, y_pred == class_id), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(y_true == class_id, y_pred == class_id), tf.float32))
        iou = tf.cond(union > 0, lambda: intersection / union, lambda: 0.0)
        ious.append(iou)
    
    return tf.reduce_mean(ious)

# Custom callbacks for better training monitoring
# class MemoryCleanupCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         """Clean up memory at the end of each epoch"""
#         gc.collect()
#         # Log memory usage
#         process = psutil.Process()
#         memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
#         print(f"\nMemory usage after epoch {epoch}: {memory_usage:.2f} MB")

class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, save_dir, num_samples=4, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.save_frequency = save_frequency  # Save every N epochs
        os.makedirs(save_dir, exist_ok=True)
        
        # Palette for official 19 trainIds (0..18); add 20th color for 'void'=19
        self.trainids_colors = [
            [128, 64, 128],   # 0: road
            [244, 35, 232],   # 1: sidewalk
            [70, 70, 70],     # 2: building
            [102, 102, 156],  # 3: wall
            [190, 153, 153],  # 4: fence
            [153, 153, 153],  # 5: pole
            [250, 170, 30],   # 6: traffic light
            [220, 220, 0],    # 7: traffic sign
            [107, 142, 35],   # 8: vegetation
            [152, 251, 152],  # 9: terrain
            [70, 130, 180],   # 10: sky
            [220, 20, 60],    # 11: person
            [255, 0, 0],      # 12: rider
            [0, 0, 142],      # 13: car
            [0, 0, 70],       # 14: truck
            [0, 60, 100],     # 15: bus
            [0, 80, 100],     # 16: train
            [0, 0, 230],      # 17: motorcycle
            [119, 11, 32],    # 18: bicycle
            [0, 0, 0],        # 19: void
        ]
        self.ignore_color = [0, 0, 0]
    
    def colorize_mask(self, mask):
        """Convert trainIds mask (0..18, 255 ignore) to RGB colors"""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        train_ids = mask.astype(np.uint8)
        unique_ids = np.unique(train_ids)
        for tid in unique_ids:
            if tid == 255:
                colored_mask[train_ids == tid] = self.ignore_color
            elif 0 <= tid < len(self.trainids_colors):
                colored_mask[train_ids == tid] = self.trainids_colors[tid]
        return colored_mask
    
    def on_epoch_end(self, epoch, logs=None):
        # Save every epoch for the first 5 epochs, then follow save_frequency
        if (epoch + 1) <= 5 or ((epoch + 1) % self.save_frequency == 0):
            try:
                saved = 0
                batch_idx = 0
                max_batches = min(8, len(self.val_generator))  # Allow 4 batches with optimizations
                
                # Temporarily disable XLA for prediction to avoid compilation issues
                try:
                    # Disable JIT compilation for prediction
                    tf.config.experimental.disable_jit()
                except:
                    pass  # JIT might already be disabled
                
                while saved < self.num_samples and batch_idx < max_batches:
                    try:
                        # Fetch a small validation batch
                        batch = self.val_generator[batch_idx]
                        if len(batch) == 3:
                            batch_images, batch_masks, batch_weights = batch
                        else:
                            batch_images, batch_masks = batch
                            batch_weights = np.ones(batch_masks.shape[:3], dtype=np.float32)
                        
                        # Use a smaller batch size for prediction to avoid memory issues
                        if batch_images.shape[0] > 2:
                            batch_images = batch_images[:2]
                            batch_masks = batch_masks[:2]
                            batch_weights = batch_weights[:2] if batch_weights is not None else None
                        
                        # Use eager execution for prediction to avoid XLA compilation
                        with tf.device('/CPU:0'):  # Force CPU execution for prediction
                            predictions = self.model(batch_images, training=False)
                            predictions = predictions.numpy()

                        take = min(self.num_samples - saved, batch_images.shape[0])
                        for i in range(take):
                            try:
                                sample_dir = os.path.join(self.save_dir, f'sample_{saved + 1}')
                                os.makedirs(sample_dir, exist_ok=True)

                                original_img = (batch_images[i] * 255).astype(np.uint8)

                                gt_mask = np.argmax(batch_masks[i], axis=-1)
                                # Reapply ignore (255) where sample weights are 0
                                if batch_weights is not None:
                                    ignore_mask = batch_weights[i] < 0.5
                                    gt_mask_vis = gt_mask.copy()
                                    gt_mask_vis[ignore_mask] = 255
                                else:
                                    gt_mask_vis = gt_mask
                                gt_colored = self.colorize_mask(gt_mask_vis)

                                pred_mask = np.argmax(predictions[i], axis=-1)
                                pred_colored = self.colorize_mask(pred_mask)

                                combined = np.hstack([original_img, gt_colored, pred_colored])

                                save_path = os.path.join(sample_dir, f'epoch_{epoch+1:03d}.png')
                                success = cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                                
                                if success:
                                    saved += 1
                                else:
                                    print(f"⚠️  Failed to save image: {save_path}")
                                    
                            except Exception as img_error:
                                print(f"⚠️  Error processing image {i}: {img_error}")
                                continue

                    except Exception as batch_error:
                        print(f"⚠️  Error processing batch {batch_idx}: {batch_error}")
                        batch_idx += 1
                        continue

                    batch_idx += 1

                if saved > 0:
                    print(f"\n📸 Saved {saved} validation visualizations for epoch {epoch+1} in {self.save_dir}")
                else:
                    print(f"\n⚠️  No validation samples available to visualize at epoch {epoch+1}")

            except Exception as e:
                print(f"⚠️  Could not save prediction visualizations: {e}")            
                traceback.print_exc()
            finally:
                # Restore XLA settings
                try:
                    # Re-enable JIT compilation for training
                    tf.config.experimental.enable_jit()
                except:
                    pass  # Ignore errors in cleanup

class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, num_classes=NUM_CLASSES, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Learning rate tracking (this is still useful since it's not automatically logged)
            if logs is not None and hasattr(self.model.optimizer, 'learning_rate'):
                current_lr = float(self.model.optimizer.learning_rate)
                logs['learning_rate'] = current_lr
                
        except Exception as e:
            print(f"⚠️  Error in DetailedMetricsCallback: {e}")

def create_callbacks(model_name, val_generator):
    """Create training callbacks"""
    callbacks = []
    
    # Create visualization directory
    viz_dir = os.path.join(MODELS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Model checkpoint - save weights only (like original trainQAT.py)
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint_epoch_{{epoch:02d}}.weights.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  # Save weights only to avoid Keras format issues
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    
    # Learning rate reduction
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    # Prediction visualization callback
    viz_callback = PredictionVisualizationCallback(
        val_generator=val_generator,
        save_dir=viz_dir,
        num_samples=4,
        save_frequency=5
    )
    callbacks.append(viz_callback)
    
    # Detailed metrics callback
    metrics_callback = DetailedMetricsCallback(
        val_generator=val_generator,
        save_frequency=1
    )
    callbacks.append(metrics_callback)
    
    # TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=TB_LOGS_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    )
    callbacks.append(tensorboard_callback)
    
    # Memory cleanup callback
    # memory_callback = MemoryCleanupCallback()
    # callbacks.append(memory_callback)
        
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
        try:
            # Try to load as full model first
            model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
            print("Loaded full model successfully!")
        except Exception as e:
            print(f"Could not load as full model: {e}")
            # Try to load as weights-only checkpoint
            if LOAD_MODEL_PATH.endswith('.weights.h5'):
                print("Attempting to load weights-only checkpoint...")
                model.load_weights(LOAD_MODEL_PATH)
                print("Loaded weights successfully!")
            else:
                print(f"Unsupported model format: {LOAD_MODEL_PATH}")
                print("Continuing with fresh model...")
    
    # Compile model
    print("\nCompiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # Use categorical crossentropy loss
        metrics=[
            "accuracy",
            dice_loss,
            mean_iou,
        ]
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    model_name = "cgra4ml_lmiinet_qat" if TRAIN_QAT else "cgra4ml_lmiinet_baseline"
    callbacks = create_callbacks(model_name, val_generator)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model (architecture + weights) at the end of training in .h5 format
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.h5")
    print(f"\nSaving final model to: {final_model_path}")
    try:
        # Save full model (architecture + weights) for CGRA4ML compatibility
        model.save(final_model_path, save_format='h5')
        print("Full model saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save full model: {e}")
        print("Saving weights only as fallback...")
        try:
            model.save_weights(final_model_path.replace('_final.h5', '_weights.h5'))
            print("Weights saved successfully!")
        except Exception as e2:
            print(f"Error saving weights: {e2}")

    print("\nTraining completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
