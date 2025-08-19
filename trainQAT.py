
#!/usr/bin/env python3
"""
LMIINet: HLS4ML-Compatible Lightweight Multiple-Information Interaction Network
==============================================================================

This script implements LMIINet architecture from the research paper:
"Efficient Semantic Segmentation via Lightweight Multiple-Information Interaction Network"
https://arxiv.org/html/2410.02224v1

Modified for full HLS4ML compatibility using only supported layers.

LMIINet achieves 72.0% mIoU on Cityscapes with only 0.72M parameters and 11.74G FLOPs.

This script implements the recommended hls4ml approach:
1. Train a baseline model in float32 (trainQAT=False)
2. Load the trained float32 model and apply QAT (trainQAT=True)

Usage:
    python trainQAT.py --trainQAT=False  # Train baseline float32 model
    python trainQAT.py --trainQAT=True   # Apply QAT to trained model

Key Features:
- HLS4ML-compatible LMIINet architecture with LFIB modules
- Global context blocks (replacing transformer blocks) for HLS4ML compatibility
- Configurable dropout layers for regularization (controlled by --use_dropout flag)
- Baseline float32 training for optimal convergence
- QAT application after float32 training completes
- Memory-efficient data pipeline
- Full FPGA compatibility using only supported layers

HLS4ML Compatibility:
- Uses only supported layers: Conv2D, DepthwiseConv2D, BatchNormalization, Activation
- GlobalAveragePooling2D, Dense, Reshape, Add, Concatenate, UpSampling2D
- No unsupported layers: Lambda, LayerNormalization, MultiHeadAttention, Permute
- All operations are hardware-friendly for FPGA deployment
- FIXED: Removed 'alpha': 'auto_po2' from quantizer config to prevent unsupported quantized_po2/quantized_relu_po2
- Uses only HLS4ML-supported quantizers: quantized_bits and quantized_relu

Key improvements:
- Combined loss function (categorical crossentropy + dice loss + focal loss)
- Enhanced data augmentation with geometric and color transformations
- Post-processing callback to clean scattered pixels
- More conservative learning rate scheduling with exponential decay
- Lower initial learning rate for stability
- FPGA-optimized LFIB modules with asymmetric and dilated convolutions
- Enhanced QAT configuration with stochastic rounding and layer-wise quantization
- Layer-wise quantization: BatchNorm layers use lower precision (8-bit) for stability
- Activation quantization with stochastic rounding for better training
- Improved learning rate scheduling with cooldown
- Global context blocks for better context understanding (HLS4ML compatible)
- ACCURACY-FOCUSED QAT CONFIGURATION: Prioritizes maximum accuracy over model size
  * Convolution layers: 32-bit (8 int bits) for maximum precision
  * BatchNorm layers: 16-bit (4-8 int bits) for stable training
  * Dense/MLP layers: 16-bit (4 int bits) for efficiency without accuracy loss
  * Output layer: 32-bit (8 int bits) for critical final predictions
  * Activations: 32-bit (16 int bits) for maximum accuracy preservation

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training tensorboard --logdir="/app/models/tb_logs" --host=0.0.0.0 --port=6006

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=True --subset_ratio=0.9 
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=True --subset_ratio=0.9 --LR=1e-5
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=140 --batch_size=3 --use_dropout=False --subset_ratio=0.9 

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=True --epochs=1 --batch_size=1 --use_dropout=True --subset_ratio=0.9 --LR=1.536e-4 --load_model=./models/qat_model_checkpoint_epoch_60.keras
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 analyze_iou.py
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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Multiply, Dropout, Dense, Add, DepthwiseConv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Reshape
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
    parser = argparse.ArgumentParser(description='HLS4ML Semantic Segmentation with QAT Toggle')
    parser.add_argument('--trainQAT', type=str, default='False', 
                       help='Enable QAT training (True/False). Default: False for float32 baseline')
    parser.add_argument('--load_model', type=str, default='', 
                       help='Path to float32 model to load for QAT training')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size for training (smaller for FPGA memory constraints)')
    parser.add_argument('--use_dropout', type=str, default='True',
                       help='Enable dropout during training (True/False). Default: True')
    parser.add_argument('--subset_ratio', type=float, default=1.0,
                       help='Fraction of dataset to use per epoch (0.1-1.0). Default: 1.0 (full dataset)')
    parser.add_argument('--LR', type=float, default=None,
                       help='Override default learning rate. If not set, uses recommended value for float32/QAT.')

    args = parser.parse_args()

    # Convert string to boolean and parse args
    TRAIN_QAT = args.trainQAT.lower() == 'true'
    LOAD_MODEL_PATH = args.load_model
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    USE_DROPOUT_DURING_TRAINING = args.use_dropout.lower() == 'true'
    SUBSET_RATIO = args.subset_ratio
    USER_LR = args.LR

    print(f"Training Configuration:")
    print(f"  QAT Training: {TRAIN_QAT}")
    print(f"  Load Model: {LOAD_MODEL_PATH if LOAD_MODEL_PATH else 'None (training from scratch)'}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Use Dropout: {USE_DROPOUT_DURING_TRAINING}")
    print(f"  Subset Ratio: {SUBSET_RATIO} ({SUBSET_RATIO*100:.1f}% of dataset)")
    print(f"  Dynamic Loss Weights: CE+Dice+Focal → CE+Focal (dice=0 by epoch {int(EPOCHS*0.7)})")

# Define constants
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 20
DATASET_DIR = '/data'
NUM_WORKERS = 4
PREFETCH_MAX = 8

DEBUG_MODE = False
# Ensure global dropout flag is always defined for import
USE_DROPOUT_DURING_TRAINING = False

MODEL_CONFIG = {
    # MobileNetV2 channels (fixed to match DeepLabV3+ MobileNetV2 ~2.7M params)
    'mv2_init': 16,
    'mv2_c1':   16,
    'mv2_c2':   24,
    'mv2_c3':   32,
    'mv2_c4':   64,
    'mv2_c5':   96,
    'mv2_c6':   160,
    'mv2_c7':   320,
    # ASPP and decoder configuration
    'aspp_filters':    128,
    'low_level_proj':  32,
    'decoder_filters': 128,
    # Use dilated convolutions in decoder refinement blocks
    'use_dilated_conv': True,
}


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


# BatchNorm configuration (accuracy-focused)
quantized_bits_batchnorm_config = create_quantized_bits_config(16, 4, True)

# Activation quantization configuration - accuracy-focused
quantized_bits_activation_config = create_quantized_bits_config(32, 16, True)

quantized_relu_config = {
    'bits': 16,
    'integer': 8,
    'use_stochastic_rounding': True
}
# Disable layout optimizer to prevent conflicts with QKeras
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})
# No class weights; losses ignore the void class (index 19) directly


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
                    # Likely already initialized; ignore
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

def get_optimized_quantization_config(layer_name, layer_type):
    """
    Get accuracy-focused quantization configuration
    Uses 32-bit by default, only reduces to 16-bit for layers that can handle it well
    """
    # Prioritize accuracy - use 32-bit unless we have strong evidence that lower precision works well
    
    if 'bn' in layer_name.lower() or 'batch' in layer_name.lower():
        if 'moving_mean' in layer_name.lower() or 'moving_variance' in layer_name.lower():
            # BatchNorm moving stats can be large and sensitive to precision
            # Use 16-bit with generous range for these critical statistics
            return create_quantized_bits_config(16, 8, True)
        else:
            # BatchNorm gamma/beta - use 16-bit as these are typically stable
            return create_quantized_bits_config(16, 4, True)
    
    elif 'dense' in layer_name.lower() or 'mlp' in layer_name.lower():
        # Dense/MLP layers - these can often work well with 16-bit
        # But use 32-bit if accuracy is critical
        return create_quantized_bits_config(16, 4, True)
    
    elif 'output' in layer_name.lower():
        # Output layer - critical for accuracy, use 32-bit
        return create_quantized_bits_config(32, 8, True)
    
    else:
        # Default to 32-bit for maximum accuracy
        # Only use 16-bit for layers that are proven to work well with it
        return create_quantized_bits_config(32, 8, True)

def lfib_module(x, filters, use_quantization=False, name_prefix='lfib', dropout_rate=0.1):
    """
    HLS4ML-compatible Lightweight Feature Interaction Bottleneck (LFIB) module
    Uses only supported layers: Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Concatenate, Dropout
    """
    if use_quantization:
        # Accuracy-focused quantization configs
        conv_config = get_optimized_quantization_config(f'{name_prefix}_conv', 'Conv2D')
        bn_config = get_optimized_quantization_config(f'{name_prefix}_bn', 'BatchNormalization')
        act_config = quantized_relu_config  # Use 32-bit for activations
        
        # Branch 1: Depth-wise separable convolution (HLS4ML supported)
        x1 = QDepthwiseConv2D(kernel_size=3, padding='same', use_bias=False,
                             depthwise_quantizer=quantized_bits(**conv_config),
                             name=f'{name_prefix}_dw1')(x)
        x1 = QBatchNormalization(gamma_quantizer=quantized_bits(**bn_config),
                               beta_quantizer=quantized_bits(**bn_config),
                               mean_quantizer=quantized_bits(**bn_config),
                               variance_quantizer=quantized_bits(**bn_config),
                               name=f'{name_prefix}_bn1')(x1)
        x1 = QActivation(quantized_relu(**act_config), name=f'{name_prefix}_act1')(x1)
        x1 = QConv2D(filters//2, kernel_size=1, padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_pw1')(x1)
        
        # Branch 2: Asymmetric convolution (1x3 + 3x1) - HLS4ML supported
        x2 = QConv2D(filters//4, kernel_size=(1, 3), padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_asym1')(x)
        x2 = QBatchNormalization(gamma_quantizer=quantized_bits(**bn_config),
                               beta_quantizer=quantized_bits(**bn_config),
                               mean_quantizer=quantized_bits(**bn_config),
                               variance_quantizer=quantized_bits(**bn_config),
                               name=f'{name_prefix}_bn2')(x2)
        x2 = QActivation(quantized_relu(**act_config), name=f'{name_prefix}_act2')(x2)
        x2 = QConv2D(filters//4, kernel_size=(3, 1), padding='same', use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_asym2')(x2)
        
        # Branch 3: Dilated convolution (HLS4ML supported)
        x3 = QConv2D(filters//4, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
                    kernel_quantizer=quantized_bits(**conv_config),
                    bias_quantizer=quantized_bits(**conv_config),
                    name=f'{name_prefix}_dilated')(x)
        
        # Concatenate branches (HLS4ML supported)
        x_concat_temp = Concatenate(name=f'{name_prefix}_concat_temp')([x1, x2])
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x_concat_temp, x3])
        
        # Final 1x1 convolution for channel adjustment (HLS4ML supported)
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
        # Branch 1: Depth-wise separable convolution (HLS4ML supported)
        x1 = DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False,
                            name=f'{name_prefix}_dw1')(x)
        x1 = BatchNormalization(name=f'{name_prefix}_bn1')(x1)
        x1 = Activation('relu', name=f'{name_prefix}_act1')(x1)
        x1 = Conv2D(filters//2, kernel_size=1, padding='same', use_bias=False,
                   name=f'{name_prefix}_pw1')(x1)
        
        # Branch 2: Asymmetric convolution (1x3 + 3x1) - HLS4ML supported
        x2 = Conv2D(filters//4, kernel_size=(1, 3), padding='same', use_bias=False,
                   name=f'{name_prefix}_asym1')(x)
        x2 = BatchNormalization(name=f'{name_prefix}_bn2')(x2)
        x2 = Activation('relu', name=f'{name_prefix}_act2')(x2)
        x2 = Conv2D(filters//4, kernel_size=(3, 1), padding='same', use_bias=False,
                   name=f'{name_prefix}_asym2')(x2)
        
        # Branch 3: Dilated convolution (HLS4ML supported)
        x3 = Conv2D(filters//4, kernel_size=3, padding='same', dilation_rate=2, use_bias=False,
                   name=f'{name_prefix}_dilated')(x)
        
        # Concatenate branches (HLS4ML supported, only two at a time)
        x_concat_temp = Concatenate(name=f'{name_prefix}_concat_temp')([x1, x2])
        x_concat = Concatenate(name=f'{name_prefix}_concat')([x_concat_temp, x3])
        
        # Final 1x1 convolution for channel adjustment (HLS4ML supported)
        x_out = Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
                      name=f'{name_prefix}_final')(x_concat)
        x_out = BatchNormalization(name=f'{name_prefix}_bn_final')(x_out)
        x_out = Activation('relu', name=f'{name_prefix}_act_final')(x_out)
    
    # Add dropout if enabled and rate > 0
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_out = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x_out)
    
    return x_out

def global_context_block(x, use_quantization=False, name_prefix='global_context', dropout_rate=0.1):
    """
    HLS4ML-compatible global context block using only supported layers
    Replaces transformer blocks with simpler global context mechanism
    """
    channels = x.shape[-1]
    
    # Global average pooling (HLS4ML supported)
    x_pooled = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    
    # Simple MLP for global context (HLS4ML supported)
    if use_quantization:
        mlp_config = get_optimized_quantization_config(f'{name_prefix}_mlp', 'Dense')
        
        x_mlp = QDense(channels // 4, use_bias=False,
                      kernel_quantizer=quantized_bits(**mlp_config),
                      bias_quantizer=quantized_bits(**mlp_config),
                      name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = QActivation(quantized_relu(**quantized_relu_config), 
                           name=f'{name_prefix}_mlp_act')(x_mlp)
        x_mlp = QDense(channels, use_bias=False,
                      kernel_quantizer=quantized_bits(**mlp_config),
                      bias_quantizer=quantized_bits(**mlp_config),
                      name=f'{name_prefix}_mlp2')(x_mlp)
    else:
        x_mlp = Dense(channels // 4, activation='relu', use_bias=False, name=f'{name_prefix}_mlp1')(x_pooled)
        x_mlp = Dense(channels, use_bias=False, name=f'{name_prefix}_mlp2')(x_mlp)
    
    # Add dropout to MLP if enabled
    if USE_DROPOUT_DURING_TRAINING and dropout_rate > 0:
        x_mlp = Dropout(dropout_rate, name=f'{name_prefix}_mlp_dropout')(x_mlp)
    
    # Reshape to spatial dimensions for broadcasting (HLS4ML supported)
    x_mlp_reshaped = Reshape((1, 1, channels), name=f'{name_prefix}_reshape_mlp')(x_mlp)
    
    # Add global context to spatial features (HLS4ML supported)
    x_out = Add(name=f'{name_prefix}_add_global')([x, x_mlp_reshaped])
    
    return x_out

def lmiinet_encoder(inputs, use_quantization=False):
    """
    LMIINet encoder with LFIB modules and transformer blocks
    Lightweight version with reduced channel dimensions
    """
    x = inputs
    
    # Initial stem convolution (stride 2: 1/2 resolution)
    if use_quantization:
        stem_config = get_optimized_quantization_config('stem_conv', 'Conv2D')
        x = QConv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stem_conv')(x)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stem_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stem_act')(x)
    else:
        x = Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False, name='stem_conv')(x)
        x = BatchNormalization(name='stem_bn')(x)
        x = Activation('relu', name='stem_act')(x)
    
    # Stage 1: 1/4 resolution (additional stride 2)
    if use_quantization:
        x = QConv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False,
                   kernel_quantizer=quantized_bits(**stem_config),
                   bias_quantizer=quantized_bits(**stem_config),
                   name='stage1_downsample')(x)
        x = QBatchNormalization(gamma_quantizer=quantized_bits(**stem_config),
                               beta_quantizer=quantized_bits(**stem_config),
                               mean_quantizer=quantized_bits(**stem_config),
                               variance_quantizer=quantized_bits(**stem_config),
                               name='stage1_downsample_bn')(x)
        x = QActivation(quantized_relu(**quantized_relu_config), name='stage1_downsample_act')(x)
    else:
        x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage1_downsample')(x)
        x = BatchNormalization(name='stage1_downsample_bn')(x)
        x = Activation('relu', name='stage1_downsample_act')(x)
    x = lfib_module(x, 32, use_quantization, 'stage1_lfib1', dropout_rate=0.05)
    x = lfib_module(x, 32, use_quantization, 'stage1_lfib2', dropout_rate=0.05)
    stage1_out = x
    
    # Stage 2: 1/8 resolution (additional stride 2)
    if use_quantization:
        x = QConv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False,
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
        x = Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage2_downsample')(x)
        x = BatchNormalization(name='stage2_downsample_bn')(x)
        x = Activation('relu', name='stage2_downsample_act')(x)
    x = lfib_module(x, 64, use_quantization, 'stage2_lfib1', dropout_rate=0.1)
    x = lfib_module(x, 64, use_quantization, 'stage2_lfib2', dropout_rate=0.1)
    x = lfib_module(x, 64, use_quantization, 'stage2_lfib3', dropout_rate=0.1)
    stage2_out = x
    
    # Stage 3: 1/16 resolution (additional stride 2)
    if use_quantization:
        x = QConv2D(128, kernel_size=3, strides=2, padding='same', use_bias=False,
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
        x = Conv2D(128, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage3_downsample')(x)
        x = BatchNormalization(name='stage3_downsample_bn')(x)
        x = Activation('relu', name='stage3_downsample_act')(x)
    x = lfib_module(x, 128, use_quantization, 'stage3_lfib1', dropout_rate=0.15)
    x = lfib_module(x, 128, use_quantization, 'stage3_lfib2', dropout_rate=0.15)
    x = global_context_block(x, use_quantization=use_quantization, name_prefix='stage3_global', dropout_rate=0.15)
    stage3_out = x
    
    # Stage 4: 1/32 resolution (additional stride 2)
    if use_quantization:
        x = QConv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False,
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
        x = Conv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False, name='stage4_downsample')(x)
        x = BatchNormalization(name='stage4_downsample_bn')(x)
        x = Activation('relu', name='stage4_downsample_act')(x)
    x = lfib_module(x, 256, use_quantization, 'stage4_lfib1', dropout_rate=0.2)
    x = lfib_module(x, 256, use_quantization, 'stage4_lfib2', dropout_rate=0.2)
    x = global_context_block(x, use_quantization=use_quantization, name_prefix='stage4_global', dropout_rate=0.2)
    stage4_out = x
    
    return stage1_out, stage2_out, stage3_out, stage4_out

def lmiinet_decoder(stage1_out, stage2_out, stage3_out, stage4_out, use_quantization=False):
    """
    LMIINet decoder with boundary detail information fusion
    Lightweight version with reduced channel dimensions
    """
    # Start from stage4 (1/32 resolution)
    x = stage4_out

    # Upsample to stage3 (1/16) - stage3_out should be 1/16 resolution
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up1')(x)
    # Ensure spatial dimensions match before concatenation
    x = Concatenate(name='concat1')([x, stage3_out])
    x = lfib_module(x, 128, use_quantization, 'dec1_lfib', dropout_rate=0.15)

    # Upsample to stage2 (1/8) - stage2_out should be 1/8 resolution
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up2')(x)
    # Ensure spatial dimensions match before concatenation
    x = Concatenate(name='concat2')([x, stage2_out])
    x = lfib_module(x, 64, use_quantization, 'dec2_lfib', dropout_rate=0.1)

    # Upsample to stage1 (1/4) - stage1_out should be 1/4 resolution
    x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up3')(x)
    # Ensure spatial dimensions match before concatenation
    x = Concatenate(name='concat3')([x, stage1_out])
    x = lfib_module(x, 32, use_quantization, 'dec3_lfib', dropout_rate=0.05)

    # Final upsampling to full resolution (1/4 -> 1/1)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear', name='up_final')(x)

    # Final convolution for output
    if use_quantization:
        output_config = get_optimized_quantization_config('output_conv', 'Conv2D')
        x = QConv2D(NUM_CLASSES, kernel_size=1, padding='same', activation='softmax',
                   kernel_quantizer=quantized_bits(**output_config),
                   bias_quantizer=quantized_bits(**output_config),
                   name='output_conv')(x)
    else:
        x = Conv2D(NUM_CLASSES, kernel_size=1, padding='same', activation='softmax', name='output_conv')(x)

    return x


def create_segmentation_model(use_quantization=False):
    """
    Create LMIINet model based on the research paper
    Lightweight Multiple-Information Interaction Network for real-time semantic segmentation

    Args:
        use_quantization: Whether to use QKeras quantized layers
    """
    tf.keras.backend.clear_session()

    # Input
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name="input_image")

    # LMIINet encoder
    stage1_out, stage2_out, stage3_out, stage4_out = lmiinet_encoder(
        inputs, use_quantization
    )

    # LMIINet decoder
    outputs = lmiinet_decoder(
        stage1_out, stage2_out, stage3_out, stage4_out, use_quantization
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LMIINet")
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
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                    A.MotionBlur(blur_limit=4, p=0.15),
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
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load corresponding mask with robust Cityscapes handling
            # 1) Prefer *_labelTrainIds.png (already 0..18 with 255 ignore)
            mask_name_tids = image_path.name.replace(
                "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
            )
            mask_path_tids = self.mask_dir / image_path.parent.name / mask_name_tids

            mask = None
            if mask_path_tids.exists():
                mask = cv2.imread(str(mask_path_tids), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not load mask: {mask_path_tids}")
            else:
                # 2) Fallback to *_labelIds.png (34-class)
                mask_name_lids = image_path.name.replace(
                    "_leftImg8bit.png", "_gtFine_labelIds.png"
                )
                mask_path_lids = self.mask_dir / image_path.parent.name / mask_name_lids
                mask = cv2.imread(str(mask_path_lids), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not load mask: {mask_path_lids}")

            # Standard Cityscapes mapping to 20 classes: 19 trainIds (0..18) + 19th 'void' class index=19.
            # If we already
            # have labelTrainIds, use as-is. Otherwise map labelIds to trainIds;
            # all non-eval ids go to 19 (void) instead of 255.
            if not mask_path_tids.exists():
                # Map labelIds (34) to our 20-class scheme: 19 trainIds + void=19.
                # Custom merges requested:
                # - polegroup (18) → pole (5)
                # - caravan (29), trailer (30) → car (13)
                lid_to_tid = {
                    7: 0,
                    8: 1,
                    11: 2,
                    12: 3,
                    13: 4,
                    17: 5,
                    18: 5,
                    19: 6,
                    20: 7,
                    21: 8,
                    22: 9,
                    23: 10,
                    24: 11,
                    25: 12,
                    26: 13,
                    27: 14,
                    28: 15,
                    31: 16,
                    32: 17,
                    33: 18,
                    29: 13,
                    30: 13,
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

            # One-hot: map 255 (ignore) to void index 19 so it occupies a dedicated channel
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

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_paths = self.image_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []

        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)

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
                if (
                    hasattr(clean_layer, "weights")
                    and hasattr(trained_layer, "weights")
                    and len(clean_layer.weights) > 0
                    and len(trained_layer.weights) > 0
                ):

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
                                print(
                                    f"Skipping {clean_layer.name}: weight shapes incompatible"
                                )
                        else:
                            print(
                                f"Skipping {clean_layer.name}: different number of weights"
                            )
                    except Exception as e:
                        print(f"Error copying weights for {clean_layer.name}: {str(e)}")

        print(f"Successfully copied weights for {copied_count} layers")

    print("Clean model created for HLS4ML export (no dropout layers)")
    return clean_model


def create_data_generators():
    """Create training and validation data generators"""
    train_image_dir = os.path.join(DATASET_DIR, "leftImg8bit", "train")
    val_image_dir = os.path.join(DATASET_DIR, "leftImg8bit", "val")
    train_mask_dir = os.path.join(DATASET_DIR, "gtFine", "train")
    val_mask_dir = os.path.join(DATASET_DIR, "gtFine", "val")

    train_generator = CityscapesDataGenerator(
        train_image_dir,
        train_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=True,
    )

    val_generator = CityscapesDataGenerator(
        val_image_dir,
        val_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=False,
        augment=False,
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
        if layer_name.endswith("_qconv"):
            float32_name = layer_name.replace("_qconv", "_conv")
        elif layer_name.endswith("_qbn"):
            float32_name = layer_name.replace("_qbn", "_bn")
        else:
            float32_name = layer_name

        if float32_name in float32_layers:
            float32_layer = float32_layers[float32_name]

            try:
                # Transfer weights for compatible layers
                if hasattr(qat_layer, "set_weights") and hasattr(
                    float32_layer, "get_weights"
                ):
                    weights = float32_layer.get_weights()
                    if len(weights) > 0:
                        qat_layer.set_weights(weights)
                        print(f"Transferred weights: {float32_name} -> {layer_name}")
            except Exception as e:
                print(f"Could not transfer weights for {layer_name}: {e}")

    return qat_model


# Removed convert_numpy_to_python function - no longer needed with TensorFlow's built-in methods

# Removed custom TrainingStateCallback - using TensorFlow's built-in CSVLogger instead
# This provides better integration with TensorFlow's ecosystem and automatic CSV logging


# Add these loss functions after the imports
def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for semantic segmentation - measures overlap between predicted and ground truth masks
    
    FUNCTIONALITY:
    - Calculates intersection over union (IoU) between predictions and ground truth
    - Dice coefficient = (2 * intersection) / (sum_true + sum_pred)
    - Loss = 1 - dice_coefficient (lower is better)
    
    BENEFITS:
    - Handles class imbalance better than crossentropy
    - Reduces scattered pixel predictions
    - Focuses on shape and boundary accuracy
    - Works well for small objects and rare classes
    
    PARAMETERS:
    - smooth: Prevents division by zero (default: 1e-6)
    """
    # Compute per-class Dice over classes 0..18 (exclude void=19), unweighted
    y_true_19 = y_true[..., :19]
    y_pred_19 = y_pred[..., :19]
    y_true_clipped = K.clip(y_true_19, 0.0, 1.0)
    y_pred_clipped = K.clip(y_pred_19, 0.0, 1.0)
    intersection = K.sum(y_true_clipped * y_pred_clipped, axis=[0, 1, 2])
    sums = K.sum(y_true_clipped, axis=[0, 1, 2]) + K.sum(y_pred_clipped, axis=[0, 1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (sums + smooth)
    return K.mean(1.0 - dice_per_class)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance - downweights easy examples and focuses on hard ones
    
    FUNCTIONALITY:
    - Modifies crossentropy to focus on hard-to-classify examples
    - pt = probability of correct classification
    - focal_loss = -alpha * (1-pt)^gamma * log(pt)
    
    BENEFITS:
    - Reduces scattered pixel predictions by focusing on hard examples
    - Handles class imbalance by downweighting easy background pixels
    - Improves boundary accuracy and small object detection
    - Prevents model from being dominated by background classes
    
    PARAMETERS:
    - alpha: Weighting factor for rare classes (default: 0.25)
    - gamma: Focusing parameter - higher values focus more on hard examples (default: 2.0)
    """
    # Focal loss on classes 0..18; void excluded
    y_true_19 = y_true[..., :19]
    y_pred_19 = y_pred[..., :19]
    y_true_safe = K.stop_gradient(y_true_19)
    y_pred_safe = K.clip(y_pred_19, K.epsilon(), 1.0 - K.epsilon())
    ce = -y_true_safe * K.log(y_pred_safe)
    pt = K.sum(y_true_safe * y_pred_safe, axis=-1, keepdims=True)
    modulating = K.pow(1.0 - pt, gamma)
    focal = modulating * ce
    return K.mean(K.sum(focal, axis=-1))


# def weighted_combined_loss(y_true, y_pred):
#     """
#     Combined categorical crossentropy, dice loss, and focal loss
#     Best of all three worlds for class imbalance and segmentation
#     Weights are updated dynamically by DynamicLossWeightCallback
#     """
#     global ce_weight, dice_weight, focal_weight
#     # Categorical crossentropy over classes 0..18 (exclude void)
#     y_true_19 = y_true[..., :19]
#     y_pred_19 = y_pred[..., :19]
#     y_pred_safe = K.clip(y_pred_19, K.epsilon(), 1.0 - K.epsilon())
#     ce_per_class = -y_true_19 * K.log(y_pred_safe)
#     ce = K.mean(K.sum(ce_per_class, axis=-1))
#     dice = dice_loss(y_true, y_pred)
#     focal = focal_loss(y_true, y_pred)
#     # Dynamic weights updated by callback
#     return ce_weight * ce + dice_weight * dice + focal_weight * focal


# Add custom metrics for better monitoring
def set_dynamic_loss_weights(
    epoch,
    total_epochs,
    transition_point=0.7,
    initial_ce_weight=0.1,
    initial_dice_weight=0.8,
    initial_focal_weight=0.1,
):
    """Stateless helper to set global loss weights based on epoch progress.

    Avoids any callback binding issues by using a LambdaCallback.
    """
    global ce_weight, dice_weight, focal_weight
    progress = float(epoch) / float(total_epochs) if total_epochs else 0.0
    if progress <= transition_point:
        transition_progress = progress / transition_point if transition_point else 1.0
        ce_weight = initial_ce_weight + (0.9 - initial_ce_weight) * transition_progress
        dice_weight = initial_dice_weight * (1.0 - transition_progress)
        focal_weight = initial_focal_weight
    else:
        ce_weight = 0.9
        dice_weight = 0.0
        focal_weight = 0.1
    total_weight = ce_weight + dice_weight + focal_weight
    ce_weight /= total_weight
    dice_weight /= total_weight
    focal_weight /= total_weight
    try:
        print(
            f"Epoch {epoch+1}/{total_epochs} - Loss weights: CE={ce_weight:.3f}, Dice={dice_weight:.3f}, Focal={focal_weight:.3f}"
        )
    except Exception:
        pass


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
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))

        # Match the correct calculate_iou logic
        iou = tf.where(union > 0, intersection / union,
                       tf.where(intersection == 0, 1.0, 0.0))  # 1.0 if both intersection and union are 0
        ious.append(iou)

    result = tf.reduce_mean(ious)

    # Clean up intermediate tensors
    del ious, intersection, union, iou
    return result


class GroupClassIoUMetric(tf.keras.metrics.Metric):
    """Streaming IoU over an epoch for a group of classes (treated as one).

    Returns 0.0 if the class never appears in the epoch (no-union case).
    """
    def __init__(self, target_class_ids, name="group_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_class_ids = tf.constant(list(target_class_ids), dtype=tf.int32)
        self.total_intersection = self.add_weight(name=f"{name}_intersection", initializer="zeros", dtype=tf.float32)
        self.total_union = self.add_weight(name=f"{name}_union", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_ids = tf.argmax(y_true, axis=-1, output_type=tf.int32)

        pred_pos = tf.reduce_any(tf.equal(tf.expand_dims(y_pred_ids, -1), self.target_class_ids), axis=-1)
        true_pos = tf.reduce_any(tf.equal(tf.expand_dims(y_true_ids, -1), self.target_class_ids), axis=-1)

        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_pos, pred_pos), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_pos, pred_pos), tf.float32))

        self.total_intersection.assign_add(intersection)
        self.total_union.assign_add(union)

    def result(self):
        return tf.where(self.total_union > 0.0, self.total_intersection / self.total_union, 0.0)

    def reset_state(self):
        self.total_intersection.assign(0.0)
        self.total_union.assign(0.0)


def precision_metric(y_true, y_pred):
    """
    Precision metric for semantic segmentation - measures accuracy of predictions
    
    FUNCTIONALITY:
    - Calculates True Positives / (True Positives + False Positives)
    - Measures how many predicted pixels of each class were actually correct
    - Precision = TP / (TP + FP) for each class, then averaged
    
    BENEFITS:
    - Shows if model is making false positive predictions (low precision = over-prediction)
    - Important for background classes - ensures they're not being over-predicted
    - Helps identify if model is too aggressive in predictions
    - Good for detecting scattered pixel issues (low precision = false scattered pixels)
    
    INTERPRETATION:
    - High precision: Model's predictions are mostly correct
    - Low precision: Model makes many false positive predictions (over-segmentation)
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Calculate precision for each class
    precisions = []
    for class_id in range(NUM_CLASSES):
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
        precision = tf.where(tp + fp > 0, tp / (tp + fp), 0.0)
        precisions.append(precision)
    
    result = tf.reduce_mean(precisions)
    
    # Clean up intermediate tensors
    del precisions, tp, fp, precision
    return result

def recall_metric(y_true, y_pred):
    """
    Recall metric for semantic segmentation - measures completeness of predictions
    
    FUNCTIONALITY:
    - Calculates True Positives / (True Positives + False Negatives)
    - Measures how many actual pixels of each class were correctly predicted
    - Recall = TP / (TP + FN) for each class, then averaged
    
    BENEFITS:
    - Shows if model is missing objects/classes (low recall = missing predictions)
    - Important for rare classes - ensures they're not being ignored
    - Helps identify if model is too conservative in predictions
    - Good for detecting scattered pixel issues (low recall = missing scattered pixels)
    
    INTERPRETATION:
    - High recall: Model finds most actual instances of each class
    - Low recall: Model misses many instances (under-segmentation)
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Calculate recall for each class
    recalls = []
    for class_id in range(NUM_CLASSES):
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.not_equal(y_pred, class_id)), tf.float32))
        recall = tf.where(tp + fn > 0, tp / (tp + fn), 0.0)
        recalls.append(recall)
    
    result = tf.reduce_mean(recalls)
    
    # Clean up intermediate tensors
    del recalls, tp, fn, recall
    return result

def f1_score_metric(y_true, y_pred):
    """
    F1 score metric for semantic segmentation - harmonic mean of precision and recall
    
    FUNCTIONALITY:
    - Calculates harmonic mean of precision and recall: F1 = 2 * (P * R) / (P + R)
    - Balances both precision and recall in a single metric
    - F1 = 2 * (precision * recall) / (precision + recall)
    
    BENEFITS:
    - Single metric that considers both false positives and false negatives
    - Good for imbalanced datasets - penalizes both over and under-prediction
    - More robust than accuracy for semantic segmentation
    - Helps identify optimal balance between precision and recall
    
    INTERPRETATION:
    - High F1: Good balance between precision and recall
    - Low F1: Either low precision, low recall, or both
    - F1 = 1.0: Perfect predictions (rare in practice)
    """
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = tf.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0.0)
    
    # Clean up intermediate tensors
    del precision, recall
    return f1


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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
                import traceback
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


def train_model():
    """Main training function"""
    print(f"Starting training with QAT={'Enabled' if TRAIN_QAT else 'Disabled'}")
    print(f"Memory usage at start: {get_memory_usage():.2f} MB")

    # Create data generators
    train_generator, val_generator = create_data_generators()

    # Create models directory if it doesn't exist
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)

    # Keras recommended approach for model loading and training continuation
    initial_epoch = 0
    model = None
    model_prefix = "qat_model" if TRAIN_QAT else "float32_model"

    # Check for existing checkpoints
    latest_checkpoint = None

    # Look for the most recent checkpoint
    if not LOAD_MODEL_PATH:
        checkpoint_pattern = os.path.join(model_dir, f'{model_prefix}_checkpoint_epoch_*.keras')
        checkpoints = glob.glob(checkpoint_pattern)

        if checkpoints:
            # Sort by modification time to get the latest
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"\nFound latest checkpoint: {latest_checkpoint}")

            # Extract epoch number from filename
            try:
                epoch_part = latest_checkpoint.split('checkpoint_epoch_')[1].split('.')[0]
                initial_epoch = int(epoch_part)
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not extract epoch number, starting from epoch 0")
                initial_epoch = 0

    # TensorFlow automatically handles training history through model.fit()
    # The history object is returned by model.fit() and contains all training metrics
    training_history = None
    print(f"Training history will be managed by TensorFlow's built-in history object")

    print(f"Final initial_epoch determined: {initial_epoch}")

    # Model loading strategy following Keras best practices

    if LOAD_MODEL_PATH:
        # Load specific model path
        print(f"Loading model from specified path: {LOAD_MODEL_PATH}")

        # Extract epoch number from checkpoint filename if present
        if 'checkpoint_epoch_' in LOAD_MODEL_PATH:
            try:
                epoch_part = LOAD_MODEL_PATH.split('checkpoint_epoch_')[1].split('.')[0]
                initial_epoch = int(epoch_part)
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not extract epoch number, starting from epoch 0")
                initial_epoch = 0

        try:
            if TRAIN_QAT and 'float32_model' in LOAD_MODEL_PATH:
                # Load float32 model and convert to QAT
                print("Loading float32 model and converting to QAT...")
                try:
                    float32_model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
                except Exception as e:
                    print(f"Failed to load full model, trying weights-only: {e}")
                    # Always reconstruct architecture before loading weights
                    float32_model = create_segmentation_model(use_quantization=False)
                    float32_model.load_weights(LOAD_MODEL_PATH)
                model = convert_float32_to_qat(float32_model)

                # Reset epoch for QAT training
                initial_epoch = 0
                training_history = None
            else:
                # Load model directly
                try:
                    model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Failed to load full model, trying weights-only: {e}")
                    # Always reconstruct architecture before loading weights
                    if TRAIN_QAT:
                        model = create_segmentation_model(use_quantization=True)
                    else:
                        model = create_segmentation_model(use_quantization=False)
                    model.load_weights(LOAD_MODEL_PATH)
                    print("Weights loaded successfully")
        except Exception as e:
            print(f"Failed to load model from {LOAD_MODEL_PATH}: {e}")
            model = None

    elif latest_checkpoint:
        # Load from checkpoint
        try:
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            try:
                model = tf.keras.models.load_model(latest_checkpoint, compile=False)
                print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"trying weights-only: {e}")
                # Create model and load weights
                if TRAIN_QAT:
                    model = create_segmentation_model(use_quantization=True)
                else:
                    model = create_segmentation_model(use_quantization=False)
                model.load_weights(latest_checkpoint)
                print("Checkpoint weights loaded successfully")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            model = None

    # Create new model if loading failed
    if model is None:
        if TRAIN_QAT:
            print("Creating new QAT model")
            model = create_segmentation_model(use_quantization=True)
        else:
            print("Creating new float32 model")
            model = create_segmentation_model(use_quantization=False)
        initial_epoch = 0
        training_history = None

    # Compile model with appropriate learning rate and scheduler
    if USER_LR is not None:
        initial_learning_rate = USER_LR
        print(f"Using user-specified learning rate: {initial_learning_rate}")
    elif TRAIN_QAT:
        initial_learning_rate = 3e-4
        print(f"Using recommended QAT learning rate: {initial_learning_rate}")
    else:
        initial_learning_rate = 7e-4  # Lower initial learning rate for stability
        print(f"Using recommended float32 learning rate: {initial_learning_rate}")

    # Enhanced learning rate scheduler for better convergence
    if TRAIN_QAT:
        # QAT: Exponential decay with smaller steps
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20000 // BATCH_SIZE,  # Decay every 10 epoch
            decay_rate=0.8,
            staircase=True,
        )
    else:
        # Float32: Exponential decay with larger steps
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20000 // BATCH_SIZE,  # Decay every 19 epoch
            decay_rate=0.8,
            staircase=True,
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        # loss=dice_loss,  # Use dice loss only
        # loss=weighted_combined_loss,  # Use weighted combined loss (best for scattered pixels)
        loss='categorical_crossentropy',  # Use categorical crossentropy loss
        # loss=focal_loss,  # Use categorical crossentropy loss
        metrics=[
            "accuracy",
            dice_loss,
            mean_iou,
            # Streaming per-class IoUs (same names for CSV columns)
            GroupClassIoUMetric([0], name="road"),
            GroupClassIoUMetric([1], name="sidewalk"),
            GroupClassIoUMetric([6], name="traffic_light"),
            GroupClassIoUMetric([7], name="traffic_sign"),
        ],
    )
    print(f"Model compiled with learning rate: {lr_schedule}")

    model.summary()

    # Create callbacks with proper state management
    os.makedirs(model_dir, exist_ok=True)

    # Setup TensorBoard logging with separate directories for QAT and non-QAT
    if TRAIN_QAT:
        tensorboard_log_dir = os.path.join(model_dir, 'tb_logs_qat')
        print(f"Using QAT TensorBoard log directory: {tensorboard_log_dir}")
    else:
        tensorboard_log_dir = os.path.join(model_dir, 'tb_logs')
        print(f"Using float32 TensorBoard log directory: {tensorboard_log_dir}")

    try:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
    except FileExistsError:
        # If the path exists as a file, remove it and create directory
        if os.path.isfile(tensorboard_log_dir):
            os.remove(tensorboard_log_dir)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        else:
            # If it's a directory but we still get FileExistsError, try with a different name
            timestamp = int(time.time())
            tensorboard_log_dir = os.path.join(model_dir, f'tb_logs_{"qat" if TRAIN_QAT else ""}_{timestamp}')
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            print(f"Using alternative TensorBoard log directory: {tensorboard_log_dir}")
    except Exception as e:
        print(f"Warning: Could not create TensorBoard log directory: {e}")
        # Use a fallback directory
        tensorboard_log_dir = os.path.join(model_dir, f'logs_{"qat" if TRAIN_QAT else ""}')
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        print(f"Using fallback log directory: {tensorboard_log_dir}")

    # Setup visualization directories
    viz_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Enhanced callbacks with proper state management
    callbacks = [
        # Dynamic loss weight adjustment via stateless LambdaCallback to avoid symbolic binding issues
        tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: set_dynamic_loss_weights(
                epoch, EPOCHS, transition_point=0.8,
                initial_ce_weight=0.1, initial_dice_weight=0.8, initial_focal_weight=0.1
            )
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_checkpoint_epoch_{{epoch:02d}}.keras'),
            save_weights_only=True,  # Save weights only for checkpoints
            save_best_only=False,
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_best.keras'),
            save_weights_only=True,  # Save weights only for best checkpoint
            save_best_only=True,
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            verbose=1
        ),
        # # ReduceLROnPlateau callback removed - using ExponentialDecay scheduler instead
        # EarlyStopping(
        #     monitor='val_mean_iou',  # Monitor IoU instead of loss
        #     mode='max',  # Higher IoU is better
        #     patience=,
        #     verbose=1,
        #     restore_best_weights=True
        # ),
        PredictionVisualizationCallback(
            val_generator=val_generator,
            save_dir=viz_dir,
            num_samples=4,
            save_frequency=5
        ),
        DetailedMetricsCallback(
            val_generator=val_generator,
            save_frequency=5
        ),
        TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),
        MemoryCleanupCallback(),
        # Use TensorFlow's built-in CSVLogger instead of custom TrainingStateCallback
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(model_dir, f'{model_prefix}_training_log.csv'),
            separator=',',
            append=True
        ),
        # PostProcessingCallback(val_generator, os.path.join(viz_dir, 'postprocessing_comparison'))  # Removed - only for visuals, causes memory overhead
    ]

    # Train model with proper state management
    try:
        print(f"Starting training from epoch {initial_epoch}...")

        # If we have training history, we need to handle it properly
        if training_history:
            print("Resuming training with existing history")
            # Note: Keras will automatically continue from the last epoch
            # The history will be merged with new training results

        # Sequence returns (x, y, sample_weight) triples; pass it directly so Keras handles epochs correctly
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
            workers=NUM_WORKERS,
            use_multiprocessing=True if NUM_WORKERS > 1 else False,
            max_queue_size=PREFETCH_MAX
        )

        # Always save the full model (architecture + weights) at the end of training
        final_model_path = os.path.join(model_dir, f'{model_prefix}_final.keras')
        try:
            model.save(final_model_path, save_format='keras')
            print(f"Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"Error: Could not save full model: {e}")

        # Create and save clean model for HLS4ML export (without dropout)
        print("Creating clean model for HLS4ML export...")
        clean_model = create_hls4ml_export_model(trained_model=model, use_quantization=TRAIN_QAT)
        clean_model_path = os.path.join(model_dir, f'{model_prefix}_clean_for_hls4ml.h5')
        try:
            clean_model.save(clean_model_path, save_format='h5')
            print(f"Clean model (no dropout) saved to: {clean_model_path}")
        except Exception as e:
            print(f"Error: Could not save full clean model: {e}")

        # If QAT training, save quantized weights for hls4ml
        if TRAIN_QAT:
            quantized_weights_path = os.path.join(model_dir, f'{model_prefix}_quantized_weights.h5')
            model_save_quantized_weights(clean_model, quantized_weights_path)
            print(f"Quantized weights saved to: {quantized_weights_path}")
            print("Note: Quantized weights saved from clean model (without dropout) for HLS4ML compatibility")

        print("Training completed successfully!")

        # Print TensorBoard access information
        print(f"\n" + "="*60)
        print("📊 TensorBoard Monitoring Available!")
        print(f"📁 Log Directory: {tensorboard_log_dir}")
        print("🚀 Start TensorBoard with:")
        print(f"   tensorboard --logdir={tensorboard_log_dir}")
        print("="*60)

        return history

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model state...")

        # Save interrupted model (compatible format)
        interrupted_model_path = os.path.join(model_dir, f'{model_prefix}_interrupted.keras')
        try:
            model.save(interrupted_model_path, save_format='keras')
            print(f"Interrupted model saved to: {interrupted_model_path}")
        except Exception as e:
            print(f"Warning: Could not save full interrupted model, saving weights only: {e}")
            model.save_weights(interrupted_model_path)
            print(f"Interrupted weights saved to: {interrupted_model_path}")

        return None

    except Exception as e:
        print(f"Training failed with error: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("LMIINet: HLS4ML-Compatible Lightweight Multiple-Information Interaction Network")
    print("HLS4ML Semantic Segmentation Training")
    print("="*60)
    
    if not TRAIN_QAT:
        print("Phase 1: Training HLS4ML-compatible LMIINet float32 baseline model")
        print("This will create a high-quality baseline model without quantization")
        print("LMIINet architecture: LFIB modules + Global Context blocks (HLS4ML compatible)")
        print(f"Dropout: {'Enabled' if USE_DROPOUT_DURING_TRAINING else 'Disabled'} (use --use_dropout=True/False to control)")
        print("Expected performance: ~72.0% mIoU on Cityscapes with 0.72M parameters")
        print("All layers are HLS4ML-compatible for FPGA deployment")
        print("After training completes, use --trainQAT=True to apply QAT")
    else:
        print("Phase 2: Applying Quantization-Aware Training (QAT)")
        print("This will fine-tune the HLS4ML-compatible LMIINet model with quantized layers")
        if LOAD_MODEL_PATH:
            print(f"Loading baseline model from: {LOAD_MODEL_PATH}")
        else:
            print("Training QAT model from scratch (not recommended)")
    
    print("="*60)
    
    train_model()
