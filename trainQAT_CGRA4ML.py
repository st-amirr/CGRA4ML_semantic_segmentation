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

Recommended Training Schedule:
1. Train baseline float32 model:
    python trainQAT_CGRA4ML.py --trainQAT=False

2. Train QAT model (weights & activations quantized):
    python trainQAT_CGRA4ML.py --trainQAT=True --freeze_bn=True --use_dropout=True

3. (Optional) QAT with AutoKeras quantizer search:
    python trainQAT_CGRA4ML.py --trainQAT=True --use_autokeras=True --freeze_bn=True --use_dropout=True

4. For last 10 epochs, turn dropout off:
    python trainQAT_CGRA4ML.py --trainQAT=True --freeze_bn=True --use_dropout=False --epochs=<last_10_epochs>

5. For last 5 epochs, turn batchnorm freezing off:
    python trainQAT_CGRA4ML.py --trainQAT=True --freeze_bn=False --use_dropout=False --epochs=<last_5_epochs>

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

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training tensorboard --logdir="/app/models/tb_logs" --host=0.0.0.0 --port=6006

Example commands:
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=100 --batch_size=2 --use_dropout=True --subset_ratio=0.9 --LR=1e-4 --load_model /workspace/models/cgra4ml_lmiinet_baseline_checkpoint_epoch_81.weights.h5

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=100 --batch_size=2 --use_dropout=True --subset_ratio=0.9 --LR=5e-4 --load_model /workspace/models/cgra4ml_lmiinet_qat_checkpoint_epoch_88.weights.h5
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
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
import os, cv2, multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)
import threading
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, 
    Add, DepthwiseConv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, 
    Reshape, AveragePooling2D, MaxPooling2D, Multiply, Lambda
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
# Use KerasTuner for quantizer hyperparameter search instead of AutoQKeras
import keras_tuner as kt
from matplotlib.colors import ListedColormap
import tensorflow.keras.backend as K

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CGRA4ML Semantic Segmentation with QAT Toggle')
    parser.add_argument('--trainQAT', type=str, default='True', help='Train with QAT (True) or baseline (False)')
    parser.add_argument('--epochs', type=int, default=140, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--use_dropout', type=str, default='True', help='Use dropout during training')
    parser.add_argument('--subset_ratio', type=float, default=0.9, help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load existing model')
    parser.add_argument('--debug', type=str, default='False', help='Enable debug mode')
    parser.add_argument('--use_autokeras', type=str, default='False', help='Enable AutoKeras/KerasTuner search (True/False)')
    parser.add_argument('--freeze_bn', type=str, default='False', help='Freeze batch normalization layers during training (True/False)')
    args = parser.parse_args()

 # Configuration constants
TRAIN_QAT = True if args.trainQAT=='True' else False
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
USE_DROPOUT_DURING_TRAINING = True if args.use_dropout=='True' else False
SUBSET_RATIO = args.subset_ratio
LEARNING_RATE = args.LR
LOAD_MODEL_PATH = args.load_model
DEBUG_MODE = True if args.debug=='True' else False
USE_AUTOKERAS = True if args.use_autokeras=='True' else False
FREEZE_BN = True if args.freeze_bn=='True' else False

# Model constants
INPUT_HEIGHT = 640
INPUT_WIDTH = 1280
NUM_CLASSES = 20
NUM_WORKERS = 6  # Fixed number of workers for data loading

# Model configuration for CGRA4ML-optimized LMIINet
# This configuration is used throughout the model architecture to ensure consistency
# and match the original LMIINet parameter count (~2.7M parameters)
MODEL_CONFIG = {
    # MobileNetV2 channels (fixed to match DeepLabV3+ MobileNetV2 ~2.7M params)
    'mv2_init': 24,        # Initial stem channels
    'mv2_c1':   24,        # Stage 1 channels (1/4 resolution)
    'mv2_c2':   32,        # Stage 2 channels (1/8 resolution)
    'mv2_c3':   38,        # Stage 3 channels (1/16 resolution)
    'mv2_c4':   96,        # Stage 4 channels (1/32 resolution)
    'mv2_c5':   128,        # Additional stage channels (if needed)
    'mv2_c6':   160,       # Additional stage channels (if needed)
    'mv2_c7':   320,       # Additional stage channels (if needed)
    # ASPP and decoder configuration
    'aspp_filters':    128, # Global context MLP filters
    'low_level_proj':  64,  # Low-level feature projection
    'decoder_filters': 160, # Decoder refinement filters
    # Decoder block filters (parameterized)
    'dec1_filters': 256,
    'dec2_filters': 256,
    'dec3_filters': 256,
}

# Scaled-down configuration for KerasTuner hyperparameter search
# Uses smaller channels to speed up search while maintaining architectural patterns
TUNER_CONFIG = {
    # Scaled-down MobileNetV2 channels for faster hyperparameter search
    'mv2_init': 8,         # Initial stem channels (scaled down)
    'mv2_c1':   8,         # Stage 1 channels (1/4 resolution)
    'mv2_c2':   12,        # Stage 2 channels (1/8 resolution)
    'mv2_c3':   16,        # Stage 3 channels (1/16 resolution)
    'mv2_c4':   32,        # Stage 4 channels (1/32 resolution)
    'mv2_c5':   48,        # Additional stage channels (if needed)
    'mv2_c6':   80,        # Additional stage channels (if needed)
    'mv2_c7':   160,       # Additional stage channels (if needed)
    # ASPP and decoder configuration (scaled down)
    'aspp_filters':    64,  # Global context MLP filters
    'low_level_proj':  21,  # Low-level feature projection
    'decoder_filters': 80,  # Decoder refinement filters
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

# Quantizer helpers for QKeras
def q_w(i_bits=1):
    return f"quantized_bits(8, {i_bits}, 1)"  # (total=8, integer=i_bits, signed)

def q_a_relu(maxv=6):
    return f"quantized_relu(8, {maxv})"

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
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(n_cores // 2)
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
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)],
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
    
    # print(f"Eager execution enabled: {tf.executing_eagerly()}")
    try:
        tf.debugging.set_log_device_placement(debug_mode)
    except Exception:
        pass

# Call safe TF configuration early
safe_configure_tensorflow(DEBUG_MODE)


# We'll use KerasTuner to search quantizer bit-widths for weights and activations.
# The tuner build function is defined below and used only when a search is requested.

def build_q_model(hp, input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES):
    """Build a scaled-down LMIINet model with tunable quantization bit-widths.

    Uses the existing encoder/decoder functions with TUNER_CONFIG for faster hyperparameter search
    while maintaining the same architectural patterns as the full model.
    """
    # Choose bitwidths with KerasTuner
    w_bits = hp.Choice("w_bits", [2, 3, 4, 6, 8], default=8)
    a_bits = hp.Choice("a_bits", [2, 3, 4, 6, 8], default=8)
    
    print(f"Building model with w_bits={w_bits}, a_bits={a_bits}, input_shape={input_shape}")

    # Create model using existing functions with scaled-down TUNER_CONFIG
    model = create_cgra4ml_segmentation_model(
        use_quantization=True,
        w_bits=w_bits,
        a_bits=a_bits,
        config=TUNER_CONFIG,
        input_shape=input_shape
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=["accuracy", dice_loss, mean_iou])

    return model

# Replace manual quantization config with AutoQKeras scheduler in main training loop

def cgra4ml_lfib_module(x, filters, use_quantization=False, name_prefix='lfib', dropout_rate=0.1, w_bits=None, a_bits=None):
    """
    CGRA4ML-optimized Lightweight Feature Interaction Bottleneck (LFIB) module
    Simplified for better CGRA4ML mapping with reduced branching complexity
    """
    if use_quantization:
        # Use i_bits per block type: mid/LFIB: i=0, decoder: i=1
        i_bits_w = 0 if 'lfib' in name_prefix else 1
        # Weights quantizer
        w_quant = q_w(i_bits_w)
        b_quant = q_w(0)
        # Activations: always bounded ReLU
        a_quant = q_a_relu(6)

        # Branch 1: Standard convolution
        x1 = QConv2D(filters, kernel_size=3, padding='same', use_bias=False,
            kernel_quantizer=w_quant, bias_quantizer=b_quant, name=f'{name_prefix}_conv1')(x)
        x1 = QBatchNormalization(name=f'{name_prefix}_bn1')(x1)
        x1 = QActivation(a_quant, name=f'{name_prefix}_act1')(x1)

        # Branch 2: Dilated convolution
        x2 = QConv2D(filters, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
            kernel_quantizer=w_quant, bias_quantizer=b_quant, name=f'{name_prefix}_dilated')(x)

        # Concatenate
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x1, x2])

        # Final 1x1 convolution
        x_out = QConv2D(filters, kernel_size=1, padding='same', use_bias=False,
            kernel_quantizer=w_quant, bias_quantizer=b_quant, name=f'{name_prefix}_final')(x_concat)
        x_out = QBatchNormalization(name=f'{name_prefix}_bn_final')(x_out)
        x_out = QActivation(a_quant, name=f'{name_prefix}_act_final')(x_out)
    else:
        # Branch 1: Standard convolution
        x1 = Conv2D(filters, kernel_size=3, padding='same', use_bias=False,
            name=f'{name_prefix}_conv1')(x)
        x1 = BatchNormalization(renorm=True, renorm_momentum=0.99, name=f'{name_prefix}_bn1')(x1)
        x1 = Activation('relu', name=f'{name_prefix}_act1')(x1)

        # Branch 2: Dilated convolution
        x2 = Conv2D(filters, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
            name=f'{name_prefix}_dilated')(x)

        # Simple concatenation
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x1, x2])

        # Final 1x1 convolution
        x_out = Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
            name=f'{name_prefix}_final')(x_concat)
        x_out = BatchNormalization(renorm=True, renorm_momentum=0.99, name=f'{name_prefix}_bn_final')(x_out)
        x_out = Activation('relu', name=f'{name_prefix}_act_final')(x_out)

    # Add dropout if enabled
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_out = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_out)

    return x_out

def cgra4ml_global_context_block(x, use_quantization=False, name_prefix='global_context', dropout_rate=0.1, w_bits=None, a_bits=None, config=MODEL_CONFIG):
    """
    CGRA4ML-optimized global context block
    Simplified for better CGRA4ML mapping
    """
    channels = x.shape[-1]
    
    # Global average pooling
    x_pooled = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)

    # Simple MLP for global context
    if use_quantization:
        # Use i_bits=1 for global context weights
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x_mlp = QDense(config['aspp_filters'], use_bias=False,  # Updated to use_bias=False
                        kernel_quantizer=w_quant, bias_quantizer=b_quant, name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = QActivation(a_quant, name=f'{name_prefix}_act1')(x_mlp)
        x_mlp = QDense(channels, use_bias=False,  # Updated to use_bias=False
                        kernel_quantizer=w_quant, bias_quantizer=b_quant, name=f'{name_prefix}_mlp2')(x_mlp)
        x_mlp = QActivation(a_quant, name=f'{name_prefix}_act2')(x_mlp)

    else:
        x_mlp = Dense(config['aspp_filters'], use_bias=False, name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = BatchNormalization(renorm=True, renorm_momentum=0.99, name=f'{name_prefix}_mlp1_bn')(x_mlp)
        x_mlp = Activation('relu', name=f'{name_prefix}_act1')(x_mlp)
        x_mlp = Dense(channels, use_bias=False, name=f'{name_prefix}_mlp2')(x_mlp)
        x_mlp = BatchNormalization(renorm=True, renorm_momentum=0.99, name=f'{name_prefix}_mlp2_bn')(x_mlp)
        x_mlp = Activation('relu', name=f'{name_prefix}_act2')(x_mlp)
    
    # Reshape for broadcasting
    x_mlp = Reshape((1, 1, channels), name=f'{name_prefix}_reshape')(x_mlp)
    
    # Apply global context
    x_out = Add(name=f'{name_prefix}_add')([x, x_mlp])
    
    # Add dropout if enabled
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_out = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_out)
    
    return x_out

def cgra4ml_encoder(inputs, use_quantization=False, w_bits=None, a_bits=None, config=MODEL_CONFIG):
    """
    CGRA4ML-optimized LMIINet encoder
    Simplified for better CGRA4ML mapping with efficient skip connections
    """
    # Initial stem
    if use_quantization:
        # Stem: i_bits=1 for weights
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x = QConv2D(config['mv2_init'], kernel_size=3, strides=2, padding='same', use_bias=False,  # Updated to use_bias=False
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='stem')(inputs)
        x = QBatchNormalization(name='stem_bn')(x)
        x = QActivation(a_quant, name='stem_act')(x)
    else:
        x = Conv2D(config['mv2_init'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stem')(inputs)
        x = BatchNormalization(renorm=True, renorm_momentum=0.99, name='stem_bn')(x)
        x = Activation('relu', name='stem_act')(x)
    
    # Stage 1: 1/4 resolution (additional stride 2)
    if use_quantization:
        # Downsample: i_bits=1 for weights
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x = QConv2D(config['mv2_c1'], kernel_size=3, strides=2, padding='same', use_bias=False,  # Updated to use_bias=False
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='stage1_downsample')(x)
        x = QBatchNormalization(name='stage1_downsample_bn')(x)
        x = QActivation(a_quant, name='stage1_downsample_act')(x)
    else:
        x = Conv2D(config['mv2_c1'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage1_downsample')(x)
        x = BatchNormalization(renorm=True, renorm_momentum=0.99, name='stage1_downsample_bn')(x)
        x = Activation('relu', name='stage1_downsample_act')(x)
    x = cgra4ml_lfib_module(x, config['mv2_c1'], use_quantization, 'stage1_lfib1', dropout_rate=0.05, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_lfib_module(x, config['mv2_c1'], use_quantization, 'stage1_lfib2', dropout_rate=0.05, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage1_global', dropout_rate=0.05, w_bits=w_bits, a_bits=a_bits, config=config)
    stage1_out = x
    
    # Stage 2: 1/8 resolution
    if use_quantization:
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x = QConv2D(config['mv2_c2'], kernel_size=3, strides=2, padding='same', use_bias=False,  # Updated to use_bias=False
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='stage2_downsample')(x)
        x = QBatchNormalization(name='stage2_downsample_bn')(x)
        x = QActivation(a_quant, name='stage2_downsample_act')(x)
    else:
        x = Conv2D(config['mv2_c2'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage2_downsample')(x)
        x = BatchNormalization(renorm=True, renorm_momentum=0.99, name='stage2_downsample_bn')(x)
        x = Activation('relu', name='stage2_downsample_act')(x)
    x = cgra4ml_lfib_module(x, config['mv2_c2'], use_quantization, 'stage2_lfib1', dropout_rate=0.1, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_lfib_module(x, config['mv2_c2'], use_quantization, 'stage2_lfib2', dropout_rate=0.1, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage2_global', dropout_rate=0.1, w_bits=w_bits, a_bits=a_bits, config=config)
    stage2_out = x
    
    # Stage 3: 1/16 resolution
    if use_quantization:
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x = QConv2D(config['mv2_c3'], kernel_size=3, strides=2, padding='same', use_bias=False,  # Updated to use_bias=False
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='stage3_downsample')(x)
        x = QBatchNormalization(name='stage3_downsample_bn')(x)
        x = QActivation(a_quant, name='stage3_downsample_act')(x)
    else:
        x = Conv2D(config['mv2_c3'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage3_downsample')(x)
        x = BatchNormalization(renorm=True, renorm_momentum=0.99, name='stage3_downsample_bn')(x)
        x = Activation('relu', name='stage3_downsample_act')(x)
    x = cgra4ml_lfib_module(x, config['mv2_c3'], use_quantization, 'stage3_lfib1', dropout_rate=0.15, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_lfib_module(x, config['mv2_c3'], use_quantization, 'stage3_lfib2', dropout_rate=0.15, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage3_global', dropout_rate=0.15, w_bits=w_bits, a_bits=a_bits, config=config)
    stage3_out = x
    
    # Stage 4: 1/32 resolution
    if use_quantization:
        w_quant = q_w(1)
        b_quant = q_w(0)
        a_quant = q_a_relu(6)
        x = QConv2D(config['mv2_c4'], kernel_size=3, strides=2, padding='same', use_bias=False,  # Updated to use_bias=False
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='stage4_downsample')(x)
        x = QBatchNormalization(name='stage4_downsample_bn')(x)
        x = QActivation(a_quant, name='stage4_downsample_act')(x)
    else:
        x = Conv2D(config['mv2_c4'], kernel_size=3, strides=2, padding='same', use_bias=False, name='stage4_downsample')(x)
        x = BatchNormalization(renorm=True, renorm_momentum=0.99, name='stage4_downsample_bn')(x)
        x = Activation('relu', name='stage4_downsample_act')(x)
    x = cgra4ml_lfib_module(x, config['mv2_c4'], use_quantization, 'stage4_lfib1', dropout_rate=0.2, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_lfib_module(x, config['mv2_c4'], use_quantization, 'stage4_lfib2', dropout_rate=0.2, w_bits=w_bits, a_bits=a_bits)
    x = cgra4ml_global_context_block(x, use_quantization=use_quantization, name_prefix='stage4_global', dropout_rate=0.2, w_bits=w_bits, a_bits=a_bits, config=config)
    stage4_out = x
    
    return stage1_out, stage2_out, stage3_out, stage4_out

def cgra4ml_decoder(stage1_out, stage2_out, stage3_out, stage4_out, use_quantization=False, w_bits=None, a_bits=None, config=MODEL_CONFIG):
    """
    CGRA4ML-optimized LMIINet decoder
    Simplified skip connections for better CGRA4ML mapping
    """
    # --- Gated skip fusion helper ---
    def gated_fuse(dec, skip, name):
        # Ensure gate and skip convolutions output dec.shape[-1] channels
        z = Concatenate(name=name+'_cat')([dec, skip])
        g = Conv2D(dec.shape[-1], 1, use_bias=True, name=name+'_gate1')(z)
        g = Activation('sigmoid', name=name+'_gate_sig')(g)
        s1 = Conv2D(dec.shape[-1], 1, use_bias=True, name=name+'_skip1')(skip)
        return Add(name=name+'_fuse')([
            Multiply()([g, s1]),
            Multiply()([Lambda(lambda x: 1-x)(g), dec])
        ])

    # --- Wider decoder with separable convs ---
    def sep_decoder_block(x, filters, name, use_quantization=False, w_bits=None, a_bits=None):
        if use_quantization:
            w_quant = q_w(1)
            b_quant = q_w(0)
            a_quant = q_a_relu(6)
            x = QDepthwiseConv2D(3, padding='same', use_bias=False, name=name+'_dw')(x)  # Removed kernel_quantizer
            x = QConv2D(filters, 1, use_bias=True, name=name+'_pw', kernel_quantizer=w_quant, bias_quantizer=b_quant)(x)
            x = QBatchNormalization(name=name+'_bn')(x)
            x = QActivation(a_quant, name=name+'_act')(x)
        else:
            x = DepthwiseConv2D(3, padding='same', use_bias=False, name=name+'_dw')(x)
            x = Conv2D(filters, 1, use_bias=True, name=name+'_pw')(x)
            x = BatchNormalization(renorm=True, renorm_momentum=0.99, name=name+'_bn')(x)
            x = Activation('relu', name=name+'_act')(x)
        return x

    # Start from stage4 (1/32 resolution)
    x = stage4_out

    # Upsample to stage3 (1/16) with gated skip fusion
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up1')(x)
    x = gated_fuse(x, stage3_out, 'dec1')
    x = sep_decoder_block(x, config['dec1_filters'], 'dec1', use_quantization, w_bits, a_bits)

    # --- Auxiliary head at stage3 (QAT-aware) ---
    if use_quantization:
        w_quant = q_w(1)
        b_quant = q_w(0)
        aux3 = QConv2D(NUM_CLASSES, 1, padding='same', use_bias=True, kernel_quantizer=w_quant, bias_quantizer=b_quant, name='aux3_logits')(stage3_out)
        aux3 = QActivation(q_a_relu(6), name='aux3_relu')(aux3)
    else:
        aux3 = Conv2D(NUM_CLASSES, 1, padding='same', use_bias=True, name='aux3_logits')(stage3_out)
        aux3 = Activation('relu', name='aux3_relu')(aux3)
    aux3 = UpSampling2D(size=(2,2), interpolation='bilinear', name='aux3_up1')(aux3)
    aux3 = UpSampling2D(size=(2,2), interpolation='bilinear', name='aux3_up2')(aux3)
    aux3 = Activation('softmax', name='aux3_softmax')(aux3)

    # Upsample to stage2 (1/8) with gated skip fusion
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up2')(x)
    x = gated_fuse(x, stage2_out, 'dec2')
    x = sep_decoder_block(x, config['dec2_filters'], 'dec2', use_quantization, w_bits, a_bits)

    # Upsample to stage1 (1/4) with gated skip fusion
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up3')(x)
    x = gated_fuse(x, stage1_out, 'dec3')
    x = sep_decoder_block(x, config['dec3_filters'], 'dec3', use_quantization, w_bits, a_bits)

    # Final upsampling to full resolution (1/4 -> 1/1)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear', name='up_final')(x)

    # Final convolution for output
    if use_quantization:
        w_quant = q_w(1)
        b_quant = q_w(0)
        x = QConv2D(NUM_CLASSES, kernel_size=1, padding='same', use_bias=True,
                   activation='softmax',  # Folded activation here
                   kernel_quantizer=w_quant, bias_quantizer=b_quant, name='output_conv')(x)
    else:
        x = Conv2D(NUM_CLASSES, kernel_size=1, padding='same', activation='softmax', name='output_conv')(x)
    # Return both main and aux heads
    return [x, aux3]

def create_cgra4ml_segmentation_model(use_quantization=False, w_bits=None, a_bits=None, config=MODEL_CONFIG, input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3)):
    """
    Create CGRA4ML-optimized LMIINet model
    input_shape: tuple, e.g. (height, width, channels). If None, uses default.
    """
    tf.keras.backend.clear_session()

    inputs = Input(shape=input_shape, name="input_image")

    # CGRA4ML-optimized encoder
    stage1_out, stage2_out, stage3_out, stage4_out = cgra4ml_encoder(
        inputs, use_quantization, w_bits, a_bits, config
    )

    # CGRA4ML-optimized decoder (returns [main_head, aux3_head])
    outputs = cgra4ml_decoder(
        stage1_out, stage2_out, stage3_out, stage4_out, use_quantization, w_bits, a_bits, config
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LMIINet_CGRA4ML")

    # Freeze batch normalization layers if requested
    if 'FREEZE_BN' in globals() and FREEZE_BN:
        for layer in model.layers:
            if 'batchnorm' in layer.name.lower() or 'bn' in layer.name.lower():
                if hasattr(layer, 'trainable'):
                    layer.trainable = False
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
        num_workers=NUM_WORKERS,
        prefetch_batches=8,
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
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

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
                    A.Affine(
                        scale=(1 - 0.2, 1 + 0.2),
                        rotate=(-20, 20),
                        translate_percent=(0.2, 0.2),
                        p=0.1
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
                    A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0, 0), per_channel=False, p=0.2),
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
                        num_holes_range=(1, 8),
                        hole_height_range=(4, 16),
                        hole_width_range=(4, 16),
                        fill=0,
                        fill_mask=0,
                        p=0.1,
                    ),
                    # # Random zoom (resized crop) - new augmentation
                    # A.RandomResizedCrop(
                    #     size=(self.target_height, self.target_width),
                    #     scale=(0.5, 1.0),  # Random zoom between 100% and 300%
                    #     ratio=(0.8, 1.2),
                    #     p=0.1
                    # ),
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
                    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 14: 4, 15: 2, 16: 2, 17: 5, 18: 5, 19: 6, 20: 7,
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

            # Ensure mask is int32 and ignore label 255 is mapped to ignore_index
            ignore_index = NUM_CLASSES  # Use NUM_CLASSES as ignore index
            mask = mask.astype(np.int32)
            mask[mask == 255] = ignore_index
            # Do NOT one-hot encode here; keep as int32 for masking in loss/metrics
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
        """Generate one batch of data with parallel image loading and augmentation"""
        batch_paths = self.image_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []
        batch_masks_aux = []
        # Load sequentially; parallelism will come from model.fit(workers=...)
        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            if image is None or mask is None:
                continue
            batch_images.append(image)
            batch_masks.append(mask)
            # Downsample mask for aux output, keep as class indices
            mask_aux = cv2.resize(mask, (self.target_width // 4, self.target_height // 4), interpolation=cv2.INTER_NEAREST)
            batch_masks_aux.append(mask_aux)

        return np.array(batch_images), {
            'output_conv': np.array(batch_masks),
            'aux3_softmax': np.array(batch_masks_aux)
        }

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for semantic segmentation - measures overlap between predicted and ground truth masks
    """
    # y_true: (batch, H, W), int32, with ignore_index
    # y_pred: (batch, H, W, num_classes), float32
    ignore_index = NUM_CLASSES
    mask = tf.not_equal(y_true, ignore_index)  # shape (batch, H, W)
    y_true = tf.where(mask, y_true, 0)  # set ignore pixels to background for one-hot
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, NUM_CLASSES)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_onehot = tf.cast(y_true_onehot, tf.float32)
    mask = tf.cast(mask, tf.float32)
    # Only consider non-ignored pixels
    intersection = tf.reduce_sum(y_true_onehot * y_pred * mask[..., None], axis=[0, 1, 2])
    sums = tf.reduce_sum(y_true_onehot * mask[..., None], axis=[0, 1, 2]) + tf.reduce_sum(y_pred * mask[..., None], axis=[0, 1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (sums + smooth)
    return tf.reduce_mean(1.0 - dice_per_class)

def mean_iou(y_true, y_pred):
    """
    Mean IoU metric for semantic segmentation - only considers classes present in ground truth (excludes void=19)
    """
    # y_true: (batch, H, W), int32, with ignore_index
    # y_pred: (batch, H, W, num_classes), float32
    ignore_index = NUM_CLASSES
    mask = tf.not_equal(y_true, ignore_index)
    y_pred = tf.argmax(y_pred, axis=-1)
    ious = []
    for class_id in range(NUM_CLASSES):
        true_mask = tf.logical_and(tf.equal(y_true, class_id), mask)
        pred_mask = tf.logical_and(tf.equal(y_pred, class_id), mask)
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
        iou = tf.where(union > 0, intersection / union, tf.constant(float('nan')))
        ious.append(iou)
    ious = tf.stack(ious)
    valid_ious = tf.boolean_mask(ious, ~tf.math.is_nan(ious))
    result = tf.reduce_mean(valid_ious)
    return result

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
                        # Handle dict mask format from generator
                        if isinstance(batch, tuple) and isinstance(batch[1], dict):
                            batch_images = batch[0]
                            batch_masks_dict = batch[1]
                            batch_masks = batch_masks_dict['output_conv']
                            # Optionally: batch_masks_aux = batch_masks_dict['aux3_softmax']
                            batch_weights = np.ones(batch_masks.shape[:3], dtype=np.float32)
                        elif len(batch) == 3:
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
                            if isinstance(predictions, list):
                                predictions = predictions[0]  # Use main output
                            predictions = predictions.numpy()

                        take = min(self.num_samples - saved, batch_images.shape[0])
                        for i in range(take):
                            try:
                                sample_dir = os.path.join(self.save_dir, f'sample_{saved + 1}')
                                os.makedirs(sample_dir, exist_ok=True)

                                original_img = (batch_images[i] * 255).astype(np.uint8)




                                # Defensive shape checks for mask arrays
                                gt_mask_arr = batch_masks[i]
                                pred_arr = predictions[i]
                                # If mask is not at least 2D, skip
                                if gt_mask_arr.ndim < 2 or pred_arr.ndim < 2:
                                    print(f"⚠️  Skipping sample {i}: mask or prediction is not at least 2D (shape: {gt_mask_arr.shape}, {pred_arr.shape})")
                                    continue
                                # If mask is one-hot (HxWxC), convert to class indices; if already class indices (HxW), use as-is
                                if gt_mask_arr.ndim == 3 and gt_mask_arr.shape[-1] == len(self.trainids_colors):
                                    gt_mask = np.argmax(gt_mask_arr, axis=-1)
                                elif gt_mask_arr.ndim == 2:
                                    gt_mask = gt_mask_arr
                                else:
                                    print(f"⚠️  Skipping sample {i}: GT mask shape not recognized: {gt_mask_arr.shape}")
                                    continue
                                # Same for predictions: if logits/probabilities, convert; if already class indices, use as-is
                                if pred_arr.ndim == 3 and pred_arr.shape[-1] == len(self.trainids_colors):
                                    pred_mask = np.argmax(pred_arr, axis=-1)
                                elif pred_arr.ndim == 2:
                                    pred_mask = pred_arr
                                else:
                                    print(f"⚠️  Skipping sample {i}: prediction shape not recognized: {pred_arr.shape}")
                                    continue
                                # Reapply ignore (255) where sample weights are 0
                                if batch_weights is not None and batch_weights[i].shape == gt_mask.shape:
                                    ignore_mask = batch_weights[i] < 0.5
                                    gt_mask_vis = gt_mask.copy()
                                    gt_mask_vis[ignore_mask] = 255
                                else:
                                    gt_mask_vis = gt_mask
                                gt_colored = self.colorize_mask(gt_mask_vis)
                                pred_colored = self.colorize_mask(pred_mask)
                                # Defensive shape check for original_img
                                if original_img.ndim != 3 or gt_colored.ndim != 3 or pred_colored.ndim != 3:
                                    print(f"⚠️  Skipping sample {i}: image or mask not 3D (shapes: {original_img.shape}, {gt_colored.shape}, {pred_colored.shape})")
                                    continue
                                try:
                                    combined = np.hstack([original_img, gt_colored, pred_colored])
                                except Exception as e:
                                    print(f"⚠️  Error stacking images for sample {i}: {e}")
                                    continue
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
        save_best_only=False,  # Save every epoch, not just best
        save_weights_only=True,  # Save weights only to avoid Keras format issues
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    
    # Learning rate reduction
    lr_reducer = ReduceLROnPlateau(
        monitor='val_output_conv_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-5,
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    # Prediction visualization callback
    viz_callback = PredictionVisualizationCallback(
        val_generator=val_generator,
        save_dir=viz_dir,
        num_samples=6,
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

def convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG):
    """Convert a trained float32 model to QAT model by replacing layers and transferring weights."""
    print("Converting float32 model to QAT model...")
    qat_model = create_cgra4ml_segmentation_model(use_quantization=True, w_bits=w_bits, a_bits=a_bits, config=config)
    f_layers = {l.name: l for l in float32_model.layers}
    copied, skipped = 0, []

    for ql in qat_model.layers:
        fl = f_layers.get(ql.name)
        if not fl or not hasattr(ql, 'get_weights') or not hasattr(fl, 'get_weights'):
            continue
        fw, qw = fl.get_weights(), ql.get_weights()

        try:
            # exact match
            if len(fw) == len(qw) and all(f.shape == q.shape for f, q in zip(fw, qw)):
                ql.set_weights(fw); copied += 1; continue

            # float no-bias -> QAT bias (append zeros)
            if len(fw) + 1 == len(qw) and all(fw[i].shape == qw[i].shape for i in range(len(fw))):
                ql.set_weights(fw + [np.zeros_like(qw[-1])]); copied += 1; continue

            # BN renorm (float 7 tensors) -> QBN (4 tensors): take first 4 (gamma, beta, mean, var)
            if ('bn' in ql.name.lower()) and len(fw) > len(qw):
                base = fw[:len(qw)]
                if all(b.shape == q.shape for b, q in zip(base, qw)):
                    ql.set_weights(base); copied += 1; continue

            skipped.append((ql.name, [w.shape for w in fw], [w.shape for w in qw]))
        except Exception as e:
            skipped.append((ql.name, str(e)))

    print(f"Weight transfer coverage: {copied}/{len(qat_model.layers)}")
    if skipped[:5]:
        print("Examples of unmatched layers:", skipped[:5])
    return qat_model

def main():
    """Main training function"""
    print("=" * 60)
    print("CGRA4ML LMIINet Semantic Segmentation Training")
    print("=" * 60)
    print(f"Training with QAT: {TRAIN_QAT}")

    # Initialize data generators (user-specified argument order)
    train_generator = CityscapesDataGenerator(
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=True,
        num_workers=NUM_WORKERS,
        prefetch_batches=8
    )

    val_generator = CityscapesDataGenerator(
        VAL_IMAGES_DIR,
        VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=False,
        augment=False,
        num_workers=NUM_WORKERS,
        prefetch_batches=8
    )

    # PHASE 1: Baseline training (float32)
    if not TRAIN_QAT:
        print("\n[PHASE 1] Baseline float32 training mode.")
        if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading baseline model weights from: {LOAD_MODEL_PATH}")
            model = create_cgra4ml_segmentation_model(use_quantization=False)
            model.load_weights(LOAD_MODEL_PATH)
        else:
            print("Creating new baseline model.")
            model = create_cgra4ml_segmentation_model(use_quantization=False)

    # PHASE 2: QAT conversion and training
    elif TRAIN_QAT and not USE_AUTOKERAS:
        print("\n[PHASE 2] QAT conversion and training mode.")
        if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            if 'qat' in os.path.basename(LOAD_MODEL_PATH).lower():
                print(f"Detected QAT weights file: {LOAD_MODEL_PATH}")
                model = create_cgra4ml_segmentation_model(use_quantization=True)
                try:
                    model.load_weights(LOAD_MODEL_PATH)
                    print("Loaded QAT weights successfully.")
                except Exception as e:
                    print(f"Error loading QAT weights: {e}")
            else:
                print(f"Detected baseline weights file: {LOAD_MODEL_PATH}")
                float32_model = create_cgra4ml_segmentation_model(use_quantization=False)
                try:
                    float32_model.load_weights(LOAD_MODEL_PATH)
                    print("Loaded baseline weights successfully. Converting to QAT model...")
                    model = convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG)
                    # prevent later generic loader from re-loading float weights into QAT model
                    checkpoint_loaded = True
                    initial_epoch = 0
                    m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH or '')
                    if m:
                        initial_epoch = int(m.group(1))
                        print(f"Resuming QAT from baseline epoch {initial_epoch} (for logging only).")
                except Exception as e:
                    print(f"Error loading baseline weights: {e}")
        else:
            print("Creating new QAT model (no weights found).")
            model = create_cgra4ml_segmentation_model(use_quantization=True)
            # prevent later generic loader from re-loading float weights into QAT model
            checkpoint_loaded = True
            initial_epoch = 0

    # PHASE 3: QAT with AutoKeras quantization search
    elif TRAIN_QAT and USE_AUTOKERAS:
        print("\n[PHASE 3] QAT with AutoKeras/KerasTuner quantization search mode.")
        # Prepare a small subset for search (tuner accepts numpy arrays)
        x_train, y_train = next(iter(train_generator))
        num_sub = min(64, x_train.shape[0])
        x_sub = x_train[:num_sub]
        y_sub = y_train[:num_sub]
        target_h = max(32, INPUT_HEIGHT // 4)
        target_w = max(32, INPUT_WIDTH // 4)

        # Load QAT weights if available
        qat_weights_loaded = False
        if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading QAT model weights for tuner initialization from: {LOAD_MODEL_PATH}")
            qat_model = create_cgra4ml_segmentation_model(use_quantization=True, input_shape=(target_h, target_w, 3))
            try:
                qat_model.load_weights(LOAD_MODEL_PATH)
                print("Loaded QAT weights for tuner initialization.")
                qat_weights_loaded = True
            except Exception as e:
                print(f"Error loading QAT weights for tuner: {e}")
        x_sub_resized = tf.image.resize(x_sub, (target_h, target_w), method='bilinear').numpy()
        y_idx = np.argmax(y_sub, axis=-1)[..., np.newaxis].astype(np.float32)
        y_resized = tf.image.resize(y_idx, (target_h, target_w), method='nearest').numpy().squeeze(-1).astype(np.int32)
        y_sub_resized = tf.keras.utils.to_categorical(y_resized, num_classes=NUM_CLASSES)
        tune_dir = os.path.join(MODELS_DIR, f"tune_qkeras_{int(time.time())}")
        os.makedirs(tune_dir, exist_ok=True)
        tuner = kt.Hyperband(
            lambda hp: build_q_model(hp, input_shape=(target_h, target_w, 3), num_classes=NUM_CLASSES),
            objective=kt.Objective("val_mean_iou", direction="max"),
            max_epochs=20,
            directory=tune_dir,
            project_name="lmiinet",
            executions_per_trial=1,
        )
        # Optionally, set initial weights for tuner trials
        if qat_weights_loaded:
            print("Note: KerasTuner does not natively support weight transfer for each trial, but you can manually set weights in the build function if needed.")
        # Actually run the tuner search
        print("Starting KerasTuner search for optimal quantizer bit-widths...")
        tuner.search(
            x_sub_resized, y_sub_resized,
            epochs=10,
            validation_split=0.2,
            verbose=1
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)
        if best_hps:
            best_hp = best_hps[0]
            optimal_w_bits = best_hp.get('w_bits')
            optimal_a_bits = best_hp.get('a_bits')
            print(f"KerasTuner search complete. Optimal bit-widths: w_bits={optimal_w_bits}, a_bits={optimal_a_bits}")
            model = create_cgra4ml_segmentation_model(
                use_quantization=True,
                w_bits=optimal_w_bits,
                a_bits=optimal_a_bits,
                input_shape=(target_h, target_w, 3)
            )
            # Optionally, transfer weights from previous QAT model if shapes match
            if qat_weights_loaded:
                try:
                    model.load_weights(LOAD_MODEL_PATH)
                    print("Transferred QAT weights to optimal quantized model.")
                except Exception as e:
                    print(f"Could not transfer QAT weights to optimal model: {e}")
        else:
            print("AutoKeras/KerasTuner search disabled. Using default quantized model.")
            model = create_cgra4ml_segmentation_model(use_quantization=True)
    checkpoint_loaded = locals().get('checkpoint_loaded', False)
    initial_epoch = locals().get('initial_epoch', 0)

    if (not checkpoint_loaded) and LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        epoch_match = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH)
        if epoch_match:
            print(f"Detected epoch checkpoint in filename: {LOAD_MODEL_PATH}")
            try:
                model.load_weights(LOAD_MODEL_PATH)
                loaded_epoch = int(epoch_match.group(1))
                initial_epoch = loaded_epoch
                print(f"Loaded weights from epoch {loaded_epoch} successfully! Will resume from epoch {loaded_epoch + 1}.")
                checkpoint_loaded = True
            except Exception as e:
                print(f"Error loading weights from checkpoint: {e}")
        else:
            print(f"Loading existing model from: {LOAD_MODEL_PATH}")
            try:
                # Try to load as full model first
                float32_model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
                print("Loaded full model successfully!")
                if TRAIN_QAT:
                    print("Converting loaded float32 model to QAT model...")
                    model = convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG)
                    checkpoint_loaded = True
                    initial_epoch = 0
                    m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH or '')
                    if m:
                        initial_epoch = int(m.group(1))
                        print(f"Resuming QAT from baseline epoch {initial_epoch} (for logging only).")
                else:
                    model = float32_model
            except Exception as e:
                print(f"Could not load as full model: {e}")
                # Try to load as weights-only checkpoint
                if LOAD_MODEL_PATH.endswith('.weights.h5'):
                    print("Attempting to load weights-only checkpoint...")
                    # Create float32 model and load weights
                    float32_model = create_cgra4ml_segmentation_model(use_quantization=False)
                    float32_model.load_weights(LOAD_MODEL_PATH)
                    print("Loaded weights successfully!")
                    if TRAIN_QAT:
                        print("Converting loaded float32 weights to QAT model...")
                        model = convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG)
                        checkpoint_loaded = True
                        initial_epoch = 0
                        m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH or '')
                        if m:
                            initial_epoch = int(m.group(1))
                            print(f"Resuming QAT from baseline epoch {initial_epoch} (for logging only).")
                    else:
                        model = float32_model
                else:
                    print(f"Unsupported model format: {LOAD_MODEL_PATH}")
                    print("Continuing with fresh model...")
    


    # Compile model
    print("\nCompiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Use custom masked loss for both outputs
    def masked_categorical_crossentropy(y_true, y_pred):
        ignore_index = NUM_CLASSES
        mask = tf.not_equal(y_true, ignore_index)
        y_true = tf.where(mask, y_true, 0)
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, NUM_CLASSES)
        loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        mask = tf.cast(mask, tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-7)

    losses = {
        'output_conv': masked_categorical_crossentropy,
        'aux3_softmax': masked_categorical_crossentropy,
    }
    loss_weights = {
        'output_conv': 1.0,
        'aux3_softmax': 0.4,
    }
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            'output_conv': ["accuracy", dice_loss, mean_iou],
        }
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
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
        workers=NUM_WORKERS,
        max_queue_size=max(8, NUM_WORKERS * 2)
    )

    # Save final model (architecture + weights) at the end of training in .h5 format
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.h5")
    print(f"\nSaving final model to: {final_model_path}")
    try:
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
