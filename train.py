import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
import signal
import sys
import json
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
BATCH_SIZE = 8  # Reduced to 1 for memory constraints
EPOCHS = 200
NUM_CLASSES = 34
DATASET_DIR = '/data'  # Updated to match actual path
SUBSET_RATIO = 1.0  # Use 20% of dataset per epoch
NUM_WORKERS = 2  # Standard for data loading
PREFETCH_SIZE = 2  # Standard prefetch for efficiency
GRADIENT_ACCUMULATION_STEPS = 2  # No gradient accumulation needed with current setup

DEBUG_MODE = False
# Model architecture configuration
MODEL_CONFIG = {
    'init_filters': INPUT_HEIGHT // 32,  # 16
    'stage1_filters': INPUT_HEIGHT // 16, # 32
    'stage2_filters': INPUT_HEIGHT // 8,  # 64
    'stage3_filters': INPUT_HEIGHT // 4,  # 128
    'stage4_filters': INPUT_HEIGHT // 2,  # 256
}

# Toggle for dropout usage (True for training, False for HLS4ML export)
USE_DROPOUT_DURING_TRAINING = False

# Toggle for HLS4ML compatible .h5 format on final save
SAVE_HLS4ML_COMPATIBLE = False  # Enable HLS4ML compatibility for model export



# Configure TensorFlow to use CPU efficiently
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)

# Disable eager execution for better memory tracking
# tf.compat.v1.disable_eager_execution()  # Commented out to enable eager execution

# Add a check to confirm eager execution is enabled
print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Add TF memory logger
tf.debugging.set_log_device_placement(DEBUG_MODE)  # Set to True only for debugging

# Disable layout optimizer to prevent conflicts with QKeras
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,  # Disabled to prevent QKeras conflicts
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})

# Define standard quantization parameters for consistent bit width
# 32-bit precision configuration (highest precision, most resources)
quantized_bits_param_32 = {
    'bits': 32,
    'integer': 8,  # Fixed at 8 for FPGA optimization
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# 16-bit precision configuration (balanced precision and resources)
quantized_bits_param_16 = {
    'bits': 16,
    'integer': 8,  # Fixed at 8 for FPGA optimization
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# 8-bit precision configuration (lowest precision, fewest resources)
quantized_bits_param_8 = {
    'bits': 8,
    'integer': 4,  # Half of total bits for balanced range
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# Set default quantization for each component type
quantized_bits_param = quantized_bits_param_16  # Default to 16-bit
quantized_bits_param_init = quantized_bits_param_16  # Input layers
quantized_bits_param_deep = quantized_bits_param_16  # Deep layers
quantized_bits_param_bridge = quantized_bits_param_16  # Bridge block
quantized_bits_param_decoder = quantized_bits_param_16  # Decoder blocks
quantized_bits_param_boundary = quantized_bits_param_16  # Boundary enhancement
quantized_bits_param_light = quantized_bits_param_16  # Intermediate layers
quantized_bits_param_bn = quantized_bits_param_16  # Batch normalization

# Define standard ReLU quantization parameters
quantized_relu_param_32 = {
    'bits': 32,
    'integer': 8,
    'use_stochastic_rounding': True
}

quantized_relu_param_16 = {
    'bits': 16,
    'integer': 8,
    'use_stochastic_rounding': True
}

quantized_relu_param_8 = {
    'bits': 8,
    'integer': 4,
    'use_stochastic_rounding': True
}

# Set default ReLU quantization
quantized_relu_param = quantized_relu_param_16  # Default to 16-bit
quantized_relu_param_deep = quantized_relu_param_16  # Deep layers
quantized_relu_param_light = quantized_relu_param_16  # Light layers

def conv_block(x, filters, bits=16, kernel_size=3, strides=1, dilation_rate=1, name=None):
    """FPGA-optimized convolution block with QAT and configurable precision"""
    # Configure quantization for this block based on bits parameter
    qbits = {
        'bits': bits,
        'integer': min(bits//2, 8),  # Scale integer bits with total bits, max 8
        'symmetric': True,
        'alpha': 'auto_po2',
        'use_stochastic_rounding': True,
        'scale_axis': None
    }
    qparam = quantized_bits(**qbits)
    
    # Conv2D with quantization parameters
    x = QConv2D(
        filters, 
        kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding='same',
        kernel_quantizer=qparam,
        bias_quantizer=qparam,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f'{name}_conv'
    )(x)
    
    # BatchNorm with quantization parameters
    x = QBatchNormalization(
        gamma_quantizer=qparam,
        beta_quantizer=qparam,
        mean_quantizer=qparam,
        variance_quantizer=qparam,
        momentum=0.9,
        epsilon=1e-5,
        name=f'{name}_bn'
    )(x)
    
    # Activation with quantization parameters
    x = QActivation(qparam, name=f'{name}_act')(x)
    return x

def c2f_block(x, filters, bits=16, n=1, name=None, dilation_rate=1, dilation_bottleneck=1):
    """YOLOv8 C2f block with configurable precision and optional dilation"""
    # First conv with dilation - for expanded receptive field in initial processing
    x = conv_block(x, filters, bits=bits, dilation_rate=dilation_rate, name=f'{name}_conv1')
    
    # Split
    y = x
    
    # Optimized bottleneck blocks - single 3x3 conv with reduced channels
    for i in range(n):
        # Direct 3x3 conv with reduced channels and dilation
        y = conv_block(y, filters//2, bits=bits, kernel_size=3, dilation_rate=dilation_bottleneck, name=f'{name}_bottleneck{i}')
        x = tf.keras.layers.Concatenate(name=f'{name}_concat{i}')([x, y])
    
    return x

# Add SPPF module for better multi-scale feature extraction
def sppf_block(x, channels, name, bits=16):
    """Enhanced Spatial Pyramid Pooling - Fast (SPPF) with configurable precision"""
    qbits = {
        'bits': bits,
        'integer': min(bits//2, 8),
        'symmetric': True,
        'alpha': 'auto_po2',
        'use_stochastic_rounding': True,
        'scale_axis': None
    }
    # Branch 1: Fine details with residual boundary enhancement
    y1_detail = QConv2D(channels//4, kernel_size=3, strides=1, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_detail_conv1')(x)
    y1_refine = QConv2D(channels//4, kernel_size=3, strides=1, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_detail_conv2')(y1_detail)
    y1 = tf.keras.layers.Add(name=f'{name}_detail_residual')([y1_detail, y1_refine])
    
    # Branch 2: Multi-scale spatial pyramid with efficient dilation
    y2_d1 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=1, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_pyramid_d1')(x)
    y2_d2 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_pyramid_d2')(x)
    y2_d4 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=4, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_pyramid_d4')(x)
    
    # Branch 3: Global context with efficient pooling
    y3_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}_global_pool')(x)
    y3 = QConv2D(channels//4, kernel_size=1, strides=1, padding='same',
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_global_conv')(y3_pool)
    # Broadcast global features
    y3 = tf.keras.layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='nearest', name=f'{name}_broadcast')(y3)
    
    # Progressive feature fusion with parallel paths
    concat = tf.keras.layers.Concatenate(name=f'{name}_concat')([x, y1, y2_d1, y2_d2, y2_d4, y3])
    return conv_block(concat, channels, kernel_size=1, name=f'{name}_conv')

# Efficient Channel Attention Module
def channel_attention(x, bits=16, ratio=16, name=''):
    """Lightweight Squeeze-and-Excitation block with configurable precision"""
    channels = x.shape[-1]
    qbits = {
        'bits': bits,
        'integer': min(bits//2, 8),
        'symmetric': True,
        'alpha': 'auto_po2',
        'use_stochastic_rounding': True,
        'scale_axis': None
    }
    
    # Combined pooling and attention
    y = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}_gap')(x)
    y = QConv2D(channels, 1,
                kernel_quantizer=quantized_bits(**qbits),
                bias_quantizer=quantized_bits(**qbits),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'{name}_attention')(y)
    y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    
    return tf.keras.layers.Multiply(name=f'{name}_scale')([x, y])

def create_segmentation_model():
    tf.keras.backend.clear_session()
    
    # Input Block - RGB image input
    # Takes RGB images of cityscapes dataset with following specifications:
    # - Resolution: [512x1024] (reduced from original for memory efficiency)
    # - Channels: 3 (RGB)
    # - Batch dimension: B (variable batch size)
    inputs = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    
    # Initial Feature Extraction - reduces channels to 16
    # First convolution uses 8-bit quantization for resource efficiency in early layers
    # Resolution maintained: [512x1024] -> [512x1024]
    # Channels: 3 -> 16
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=8, name='init_conv')  # [B, 512, 1024, 16]
    # Initial Downsampling - halves spatial dimensions
    # Uses strided QConv2D with standardized 16-bit precision
    # Resolution: [512x1024] -> [256x512]
    # Channels maintained: 16 -> 16
    qbits_down = {
        'bits': 16,
        'integer': 8,  # Fixed at 8 for FPGA optimization
        'symmetric': True,
        'alpha': 'auto_po2',
        'use_stochastic_rounding': True,
        'scale_axis': None
    }
    x = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down),
                bias_quantizer=quantized_bits(**qbits_down),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='downsample1')(x)
    # [B, 256, 512, 16]

    ############ ---- Encoder Block 1 - First encoding stage ---- ############
    # Doubles the number of channels and includes skip connection for later decoding
    # Uses standardized 16-bit precision for consistent feature extraction
    # Resolution: [256x512] -> [128x256] after downsampling
    # Channels: 16 -> 32
    # Use double conv for better initial feature extraction
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='encoder1_1')
    
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='encoder1_2')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_encoder1')(x)
    
    skip1 = x

    x = QConv2D(MODEL_CONFIG['stage1_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down),
                bias_quantizer=quantized_bits(**qbits_down),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='encoder1_downsample')(x)
    # [B, 128, 256, 32]

    ############ ---- Encoder Block 2 - Second encoding stage ---- ############
    # Further doubles the feature channels for richer representation
    # Uses C2F block with n=2 as this is a critical feature extraction stage
    # Resolution: [128x256] -> [64x128] after downsampling
    # Channels: 32 -> 64
    x = c2f_block(x, MODEL_CONFIG['stage2_filters'], bits=16, n=2, name='encoder2', dilation_rate=2, dilation_bottleneck=1)  # Enhanced feature extraction with n=2
    
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_encoder2')(x)

    skip2 = x

    x = QConv2D(MODEL_CONFIG['stage2_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down),
                bias_quantizer=quantized_bits(**qbits_down),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='encoder2_downsample')(x)
    # [B, 64, 128, 64]

   ############ ---- Encoder Block 3 - Third encoding stage (deepest encoder) ---- ############
    # Incorporates channel attention mechanism for adaptive feature weighting
    # Last encoder block before bridge, uses standardized 16-bit precision
    # Resolution: [64x128] -> [32x64] after downsampling
    # Channels: 64 -> 128
    x = c2f_block(x, MODEL_CONFIG['stage3_filters'], bits=16, n=1, name='encoder3', dilation_rate=2, dilation_bottleneck=1)  # Moderate dilation for deep features
    x = channel_attention(x, bits=16, name='ca_encoder3')

    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.3, name='dropout_encoder3')(x)

    skip3 = x

    x = QConv2D(MODEL_CONFIG['stage3_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_16),
                bias_quantizer=quantized_bits(**quantized_bits_param_16),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='encoder3_downsample')(x)
    # [B, 32, 64, 128]

    ############ ---- Bridge Block - Network bottleneck ---- ############
    # Connects encoder and decoder paths with maximal feature channels
    # Uses C2F block with 32-bit precision for maximum feature preservation at the critical bridge
    # Resolution maintained: [32x64] -> [32x64]
    # Channels: 128 -> 256 (maximum channel depth)
    x = c2f_block(x, MODEL_CONFIG['stage4_filters'], bits=32, n=3, name='bridge', dilation_rate=2, dilation_bottleneck=1)  # Maximum feature extraction at bridge
    x = channel_attention(x, bits=32, name='ca_bridge')  # High precision for critical attention weights
    x = sppf_block(x, MODEL_CONFIG['stage4_filters'], bits=32, name='bridge_sppf')  # Multi-scale context with maximum precision
    # [B, 32, 64, 256]


    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.35, name='dropout_bridge')(x)
    
    ############ ---- Decoder Block 3 (deepest level) ---- upsampling from 256 to 128 channels ############
    # Processes the features from bridge and combines with encoder3 skip connection
    # Resolution: [32x64] -> [64x128]
    # Pre-processing of bridge features
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], kernel_size=1, name='decoder3_prep')  # [B, 32, 64, 256]
    # Use regular Conv2DTranspose instead of QConv2DTranspose for compatibility
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage3_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='decoder3_upsample')(x)
    # [B, 64, 128, 128]
    
    # Pre-process skip connection with standardized precision
    skip3_enhance = conv_block(skip3, MODEL_CONFIG['stage3_filters'], bits=16, kernel_size=3, name='skip3_boundary')
    skip3 = conv_block(skip3, MODEL_CONFIG['stage3_filters'], bits=16, kernel_size=1, name='skip3_adjust')
    skip3 = tf.keras.layers.Add(name='skip3_fusion')([skip3, skip3_enhance])
    
    x = tf.keras.layers.Concatenate(name='decoder3_concat')([x, skip3])
    # Efficient triple conv with residual for deep feature refinement
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv1')
    x_identity = x  # Store identity after first conv to match channel dimensions
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv2')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv3')
    x = tf.keras.layers.Add(name='decoder3_residual')([x, x_identity])
    x = channel_attention(x, name='decoder3_attention')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.25, name='dropout_decoder3')(x)

    ############ ---- Decoder Block 2 - upsampling from 128 to 64 channels ############
    # Processes the features from decoder3 and combines with encoder2 skip connection
    # Resolution: [64x128] -> [128x256]
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], kernel_size=1, name='decoder2_prep')  # [B, 64, 128, 128]
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage2_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='decoder2_upsample')(x)
    # [B, 128, 256, 64]
    # Pre-process skip connection to match feature complexity (16-bit for mid-level features)
    skip2 = conv_block(skip2, MODEL_CONFIG['stage2_filters'], bits=16, kernel_size=1, name='skip2_adjust')
    x = tf.keras.layers.Concatenate(name='decoder2_concat')([x, skip2])
    # Triple conv at mid-level decoder for better feature refinement
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv1')  # [B, 128, 256, 64]
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv2')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv3')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.2, name='dropout_decoder2')(x)

    ############ ---- Decoder Block 1 - upsampling from 64 to 32 channels ############
    # Processes the features from decoder2 and combines with encoder1 skip connection
    # Resolution: [128x256] -> [256x512]
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], kernel_size=1, name='decoder1_prep')  # [B, 128, 256, 64]
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage1_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='decoder1_upsample')(x)
    # [B, 256, 512, 32]
    # Pre-process skip connection with standardized precision
    skip1 = conv_block(skip1, MODEL_CONFIG['stage1_filters'], bits=16, kernel_size=1, name='skip1_adjust')
    x = tf.keras.layers.Concatenate(name='decoder1_concat')([x, skip1])
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder1_conv1')  # [B, 256, 512, 32]
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder1_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.15, name='dropout_decoder1')(x)

    ############ ---- Decoder Block 0 (final upsampling) - upsampling from 32 to 16 channels ############
    # Final decoder block that restores the original input resolution
    # Resolution: [256x512] -> [512x1024]
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder0_prep')  # [B, 256, 512, 32]
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['init_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name='decoder0_upsample')(x)
    # [B, 512, 1024, 16]
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=16, name='decoder0_conv')  # [B, 512, 1024, 16]
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=16, name='decoder0_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = tf.keras.layers.Dropout(0.1, name='dropout_decoder0')(x)
    # Enhanced output layers with boundary refinement using 32-bit precision for maximum accuracy
    # Boundary-aware feature enhancement with high precision quantization for critical edge detection
    x_boundary = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, padding='same',
                      kernel_quantizer=quantized_bits(**quantized_bits_param_32),
                      bias_quantizer=quantized_bits(**quantized_bits_param_32),
                      kernel_initializer='glorot_uniform',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      name='boundary_enhance1_conv')(x)
    x_boundary = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_32),
        beta_quantizer=quantized_bits(**quantized_bits_param_32),
        mean_quantizer=quantized_bits(**quantized_bits_param_32),
        variance_quantizer=quantized_bits(**quantized_bits_param_32),
        name='boundary_enhance1_bn'
    )(x_boundary)
    x_boundary = QActivation('relu', name='boundary_enhance1_act')(x_boundary)

    x_boundary = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, padding='same',
                      kernel_quantizer=quantized_bits(**quantized_bits_param_32),
                      bias_quantizer=quantized_bits(**quantized_bits_param_32),
                      kernel_initializer='glorot_uniform',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      name='boundary_enhance2_conv')(x_boundary)
    x_boundary = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_32),
        beta_quantizer=quantized_bits(**quantized_bits_param_32),
        mean_quantizer=quantized_bits(**quantized_bits_param_32),
        variance_quantizer=quantized_bits(**quantized_bits_param_32),
        name='boundary_enhance2_bn'
    )(x_boundary)
    
    # Merge boundary features
    x = tf.keras.layers.Concatenate(name='final_fusion')([x, x_boundary])
    
    # Final reduce block with 16-bit precision for FPGA optimization
    x = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=1, padding='same',
              kernel_quantizer=quantized_bits(**quantized_bits_param_16),
              bias_quantizer=quantized_bits(**quantized_bits_param_16),
              kernel_initializer='glorot_uniform',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
              name='final_reduce_conv')(x)
    x = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_16),
        beta_quantizer=quantized_bits(**quantized_bits_param_16),
        mean_quantizer=quantized_bits(**quantized_bits_param_16),
        variance_quantizer=quantized_bits(**quantized_bits_param_16),
        name='final_reduce_bn'
    )(x)
    x = QActivation(quantized_relu(**quantized_relu_param_16), name='final_reduce_act')(x)
    
    # Output layer with high precision
    outputs = QConv2D(NUM_CLASSES, (1, 1),
                     kernel_quantizer=quantized_bits(**quantized_bits_param_16),
                     bias_quantizer=quantized_bits(**quantized_bits_param_16),
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                     name='output')(x)
    # [B, 512, 1024, NUM_CLASSES]
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
                A.GaussNoise(var_limit=(5.0, 25.0), mean=0, p=0.2),
                
                # Basic cutout (updated for newer albumentations)
                A.CoarseDropout(
                    max_holes=4,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
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
        iou = logs.get('iou', None)
        val_loss = logs.get('val_loss', None)
        val_iou = logs.get('val_iou', None)
        # Format time as seconds
        time_str = f"{int(epoch_duration)}s"
        # Compose log line similar to terminal output
        log_line = f"{time_str} 1s/step - "
        if loss is not None:
            log_line += f"loss: {loss:.4f} - "
        if iou is not None:
            log_line += f"iou: {iou:.4f} - "
        if val_loss is not None:
            log_line += f"val_loss: {val_loss:.4f} - "
        if val_iou is not None:
            log_line += f"val_iou: {val_iou:.4f} - "
        log_line += f"lr: {lr:.6e}\n"
        # Use the same file extension logic as the main training code
        file_extension = '.h5' if SAVE_HLS4ML_COMPATIBLE else '.keras'
        log_line += f"Epoch {epoch+1}: saving model to /app/models/checkpoint_epoch_{epoch+1:02d}{file_extension}\n"
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"[TrainingLoggerCallback] Could not write to log.txt: {e}")

# Custom IoU Loss for semantic segmentation
class IoULoss(tf.keras.losses.Loss):
    def __init__(self, name='iou_loss', smooth=1e-6):
        super(IoULoss, self).__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        # Flatten the predictions and ground truth
        y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
        y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])

        # Calculate intersection and union
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=0)
        union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) - intersection + self.smooth

        # Calculate IoU for each class
        iou = (intersection + self.smooth) / union
        
        # Return negative mean IoU as loss (since we want to maximize IoU)
        return 1.0 - tf.reduce_mean(iou)

# Combo Loss combining IoU and Cross-Entropy with adaptive weighting
class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, weighting_strategy='adaptive_magnitude', name='combo_loss'):
        super(ComboLoss, self).__init__(name=name)
        self.alpha = alpha  # Base weight for IoU Loss (used in some strategies)
        self.weighting_strategy = weighting_strategy
        self.iou_loss = IoULoss()
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # Moving averages for adaptive weighting
        self.iou_ema = tf.Variable(1.0, trainable=False, name='iou_ema')
        self.ce_ema = tf.Variable(1.0, trainable=False, name='ce_ema')
        self.ema_decay = 0.9  # Exponential moving average decay

    def call(self, y_true, y_pred):
        iou_loss = self.iou_loss(y_true, y_pred)
        ce_loss = self.ce_loss(y_true, y_pred)
        
        # Handle NaN values
        iou_loss = tf.where(tf.math.is_finite(iou_loss), iou_loss, 0.0)
        ce_loss = tf.where(tf.math.is_finite(ce_loss), ce_loss, 0.0)
        
        if self.weighting_strategy == 'static':
            # Original static weighting
            alpha_iou = 0.7
            alpha_ce = 0.3
            
        elif self.weighting_strategy == 'adaptive_magnitude':
            # Adaptive weighting based on loss magnitudes
            # Update exponential moving averages
            self.iou_ema.assign(self.ema_decay * self.iou_ema + (1 - self.ema_decay) * iou_loss)
            self.ce_ema.assign(self.ema_decay * self.ce_ema + (1 - self.ema_decay) * ce_loss)
            
            # Calculate relative magnitudes
            total_magnitude = self.iou_ema + self.ce_ema + 1e-8
            # Inverse weighting: give more weight to the loss that's currently smaller
            alpha_iou = self.ce_ema / total_magnitude
            alpha_ce = self.iou_ema / total_magnitude
            
        elif self.weighting_strategy == 'confidence_based':
            # Weight based on prediction confidence
            y_pred_max = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            confidence = tf.reduce_mean(y_pred_max)
            # Higher confidence -> more weight on IoU, lower confidence -> more weight on CE
            alpha_iou = 0.5 + 0.3 * confidence  # Range: [0.5, 0.8]
            alpha_ce = 1.0 - alpha_iou
            
        elif self.weighting_strategy == 'balanced_focal':
            # Focal loss inspired weighting based on prediction difficulty
            y_pred_entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-8), axis=-1)
            avg_entropy = tf.reduce_mean(y_pred_entropy)
            # Higher entropy (more uncertain) -> more weight on CE
            # Lower entropy (more certain) -> more weight on IoU
            entropy_weight = tf.clip_by_value(avg_entropy / tf.math.log(float(NUM_CLASSES)), 0.0, 1.0)
            alpha_ce = 0.2 + 0.6 * entropy_weight  # Range: [0.2, 0.8]
            alpha_iou = 1.0 - alpha_ce
            
        else:  # default to static
            alpha_iou = 0.7
            alpha_ce = 0.3
        
        # Ensure weights are finite and sum to 1
        alpha_iou = tf.where(tf.math.is_finite(alpha_iou), alpha_iou, 0.7)
        alpha_ce = tf.where(tf.math.is_finite(alpha_ce), alpha_ce, 0.3)
        
        # Normalize to ensure they sum to 1
        total_alpha = alpha_iou + alpha_ce + 1e-8
        alpha_iou = alpha_iou / total_alpha
        alpha_ce = alpha_ce / total_alpha
        
        return alpha_iou * iou_loss + alpha_ce * ce_loss

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
    # Configure TensorFlow for optimal device placement BEFORE model creation
    # This will automatically place operations on the most suitable device
    tf.config.set_soft_device_placement(True)
    
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
            learning_rate=2e-4,  # Reduced from 1e-4 for better stability
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss=ComboLoss(weighting_strategy='adaptive_magnitude'),  # Use dynamic weighting instead of static
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

    
    # Determine file extension based on HLS4ML compatibility setting
    file_extension = '.h5' if SAVE_HLS4ML_COMPATIBLE else '.keras'
    
    # Calculate current step based on resumed epoch and steps per epoch
    steps_per_epoch = max(1, len(train_gen) // GRADIENT_ACCUMULATION_STEPS)
    current_step = initial_epoch * steps_per_epoch
    
    # Create step-aware cosine decay scheduler that handles resumption
    def step_aware_cosine_schedule(epoch, logs=None):
        """Cosine decay with warm restarts that properly handles training resumption"""
        # Calculate global step (accounts for resumed training)
        global_step = current_step + (epoch - initial_epoch) * steps_per_epoch
        
        # Cosine decay with warm restarts parameters
        initial_lr = 1e-3
        first_decay_steps = 1000
        t_mul = 2.0
        m_mul = 0.9
        alpha = 0.01  # minimum lr as fraction of initial
        
        # Find which restart cycle we're in by tracking cumulative steps
        remaining_steps = global_step
        cycle = 0
        current_cycle_length = first_decay_steps
        
        # Subtract completed cycles until we find the current cycle
        while remaining_steps >= current_cycle_length:
            remaining_steps -= current_cycle_length
            cycle += 1
            current_cycle_length = int(first_decay_steps * (t_mul ** cycle))
        
        # remaining_steps is now the position within the current cycle
        step_in_cycle = remaining_steps
        
        # Calculate learning rate for current position in cycle
        max_lr = initial_lr * (m_mul ** cycle)
        min_lr = max_lr * alpha
        
        # Cosine decay within current cycle
        cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * step_in_cycle / current_cycle_length))
        lr = min_lr + (max_lr - min_lr) * cosine_decay
        
        # Debug info (optional - remove if too verbose)
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"Debug LR: Global step={global_step}, Cycle={cycle}, Step in cycle={step_in_cycle}/{current_cycle_length}, LR={lr:.6f}")
        
        return float(lr)
    
    callbacks = [
        # Step-aware learning rate scheduler that handles resumption
        tf.keras.callbacks.LearningRateScheduler(step_aware_cosine_schedule, verbose=1),
        
        ModelCheckpoint(
            f'/app/models/best_model{file_extension}',
            save_best_only=True,
            monitor='val_iou',
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        # Save model every epoch (standard approach)
        ModelCheckpoint(
            f'/app/models/checkpoint_epoch_{{epoch:02d}}{file_extension}',
            save_freq='epoch',
            verbose=1,
            save_weights_only=True
        ),
        EarlyStopping(
            patience=40,  # Increased from 7 to allow more training time
            monitor='val_iou',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau(
        #     monitor='val_iou',
        #     mode='max',
        #     factor=0.7,  # Reduce LR by 30% when plateauing
        #     patience=10,   # Wait 3 epochs before reducing
        #     min_lr=1e-7,  # Minimum LR
        #     verbose=1,
        #     restore_best_weights=False
        # ),
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
            model.save(f'/app/models/final_model_with_dropout{file_extension}')
            print(f"Training model saved as 'final_model_with_dropout{file_extension}'")
            print("HLS4ML export skipped (dropout enabled - architectural mismatch)")
        else:
            # Training without dropout - save both training and HLS4ML models
            model.save(f'/app/models/final_model_no_dropout{file_extension}')
            print(f"Training model saved as 'final_model_no_dropout{file_extension}'")
            
            # Save HLS4ML compatible .h5 version if enabled
            if SAVE_HLS4ML_COMPATIBLE:
                model.save('/app/models/final_model_hls4ml.h5', save_format='h5')
                print("HLS4ML compatible model saved as 'final_model_hls4ml.h5'")
            
            # Create and save clean model for HLS4ML export (identical architecture)
            print("Creating HLS4ML export model...")
            clean_model = create_hls4ml_export_model(model)
            
            # Save the clean model in HLS4ML compatible format
            clean_model.save('/app/models/final_model_hls4ml_clean.h5', save_format='h5')
            
            # Save quantized weights specifically for HLS4ML
            model_save_quantized_weights(clean_model, '/app/models/final_model_hls4ml_weights.h5')
            
            # Save model configuration for HLS4ML
            config = {
                'input_shape': (INPUT_HEIGHT, INPUT_WIDTH, 3),
                'output_shape': (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES),
                'quantization_bits': {
                    'input': 16,
                    'weights': 16,
                    'bias': 16,
                    'activation': 16
                },
                'layer_precision': {
                    'init_conv': 8,  # Initial layers use 8-bit
                    'bridge': 32,    # Bridge block uses 32-bit
                    'boundary': 32,  # Boundary enhancement uses 32-bit
                    'default': 16    # Default precision for other layers
                }
            }
            
            with open('/app/models/hls4ml_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            print("Clean HLS4ML model saved as 'final_model_hls4ml_clean.h5'")
            print("Quantized weights saved as 'final_model_hls4ml_weights.h5'")
            print("HLS4ML configuration saved as 'hls4ml_config.json'")
        
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current full model...")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"/app/models/interrupted_model_{timestamp}{file_extension}"
        model.save(model_path)
        print(f"Interrupted full model saved as '{model_path}'")
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

#24/743 [..............................] - ETA: 3:05:34 - loss: 2.8576 - iou: 0.3420

