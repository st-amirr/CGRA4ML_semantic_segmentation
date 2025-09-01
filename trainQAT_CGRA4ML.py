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

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=200 --batch_size=2 --use_dropout=True --subset_ratio=0.9 --LR=1e-4 --freeze_bn=True --load_model /workspace/models/cgra4ml_lmiinet_baseline_checkpoint_epoch_81.weights.h5
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=200 --batch_size=2 --use_dropout=True --subset_ratio=0.9 --LR=1e-4 --freeze_bn=True --load_model /workspace/models/cgra4ml_lmiinet_baseline_checkpoint_epoch_05.weights.h5
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
import threading
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, 
    Add, DepthwiseConv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, 
    Reshape, AveragePooling2D, MaxPooling2D, Multiply, MultiHeadAttention, GlobalMaxPooling2D
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
import tensorflow as tf
from tensorflow.keras.models import Model
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CGRA4ML Semantic Segmentation with QAT Toggle')
    parser.add_argument('--trainQAT', type=str, default='True', help='Train with QAT (True) or baseline (False)')
    parser.add_argument('--epochs', type=int, default=140, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--use_dropout', type=str, default='True', help='Use dropout during training')
    parser.add_argument('--subset_ratio', type=float, default=0.9, help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--LR', type=float, default=7e-4, help='Learning rate')
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
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 19  # Standard Cityscapes classes (0-18), following the paper
NUM_WORKERS = 6  # Fixed number of workers for data loading

# Model configuration for CGRA4ML-optimized LMIINet
# This configuration is used throughout the model architecture to ensure consistency


MODEL_CONFIG_LMIA = {
    'encoder_filters': [24, 48, 96, 128],   # go from 24 to 128
    'decoder_filters': [128, 96, 48, 24],     # mirror down
    'transformer_heads': 8,                 # fewer heads
    'transformer_key_dim': 64,              # smaller projection
    'transformer_dropout': 0.1,
    'seghead_filters': 64,                   # Same
    'num_classes': 19                        # Correct Cityscapes classes
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
# === replace q_w / q_a_relu with this (near top of file) ===
X_BITS = 8      # activations (hw.bits_input)
K_BITS = 8      # weights     (hw.bits_weights)
B_BITS = 8      # bias        (hw.bits_bias)
QINT_BITS = 0   # integer bits, same as example

kq = f"quantized_bits({K_BITS},{QINT_BITS},False,True,1)"          # weights
bq = f"quantized_bits({B_BITS},{QINT_BITS},False,True,1)"          # bias
qr = f"quantized_relu({X_BITS},{QINT_BITS},negative_slope=0)"      # ReLU
qb = f"quantized_bits({X_BITS},{QINT_BITS},False,False,1)"         # requant (identity/clip)

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
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 12)],
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

    # Create model using LMIA config
    model = create_lmiinet(
        input_shape=input_shape,
        num_classes=num_classes,
        config=MODEL_CONFIG_LMIA
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  #loss='categorical_crossentropy',
                  loss=dice_loss,
                  metrics=[accuracy, dice_loss, mean_iou])

    return model

# Replace manual quantization config with AutoQKeras scheduler in main training loop

# CGRA4ML-Compatible Channel Shuffle Implementation
def cgra4ml_channel_shuffle(inputs, groups=2, name="shuffle"):
    """
    CGRA4ML-compatible channel shuffle using only supported operations.
    Simulates channel shuffle effect using slice and concatenate operations.
    """
   
    # Split channels into groups using slicing (CGRA4ML compatible)
    group_1 = inputs[:, :, :, :inputs.shape[-1] // groups]
    group_2 = inputs[:, :, :, inputs.shape[-1] // groups:]
    
    # Interleave groups by concatenating in reverse order
    # This creates a shuffle-like effect
    shuffled = Concatenate(axis=-1, name=f'{name}_concat')([group_2, group_1])
    
    return shuffled

# Helper function for Feature Enhancement (FE) block
def feature_enhancement(inputs, filters, name, use_quantization=True):
    """CGRA4ML-Compatible Feature Enhancement Block"""
    
    # t.Mean: Apply Global Average Pooling (GAP)
    gap = GlobalAveragePooling2D()(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]))(gap)

    # t.Max: Apply Global Max Pooling (GMP)
    gmp = GlobalMaxPooling2D()(inputs)
    gmp = Reshape((1, 1, inputs.shape[-1]))(gmp)

    # Concatenate GAP and GMP along the channel axis (axis=-1)
    x = Concatenate(axis=-1)([gap, gmp])

    if use_quantization:
        # QAT branch
        x = QConv2D(filters, (3, 3), padding='same', use_bias=True, name=f'{name}_conv_3x3', kernel_quantizer=kq, bias_quantizer=bq)(x)
        x = QBatchNormalization(name=f'{name}_bn')(x)
        a = QActivation('sigmoid', name=f'{name}_sigmoid')(x)
        if USE_DROPOUT_DURING_TRAINING:
            a = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(a)

        x2 = QConv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv_1x1', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        a2 = QActivation('sigmoid', name=f'{name}_sigmoid_a2')(x2)

        gap2 = GlobalAveragePooling2D()(inputs)
        gap2 = Reshape((1, 1, inputs.shape[-1]))(gap2)
        x3 = QConv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv2_1x1', kernel_quantizer=kq, bias_quantizer=bq)(gap2)
        x3 = QActivation('relu', name=f'{name}_relu2')(x3)
        x3 = QConv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv2_1x1_final', kernel_quantizer=kq, bias_quantizer=bq)(x3)
        b = QActivation('sigmoid', name=f'{name}_sigmoid_b')(x3)
        b2 = QConv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv_b2', kernel_quantizer=kq, bias_quantizer=bq)(b)
        output = Add()([a2, b2])
        return output
    else:
        # Float32 branch
        x = Conv2D(filters, (3, 3), padding='same', use_bias=True, name=f'{name}_conv_3x3')(x)
        x = BatchNormalization(name=f'{name}_bn', renorm=True)(x)
        a = Activation('sigmoid', name=f'{name}_sigmoid')(x)
        if USE_DROPOUT_DURING_TRAINING:
            a = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(a)

        x2 = Conv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv_1x1')(inputs)
        a2 = Activation('sigmoid', name=f'{name}_sigmoid_a2')(x2)

        gap2 = GlobalAveragePooling2D()(inputs)
        gap2 = Reshape((1, 1, inputs.shape[-1]))(gap2)
        x3 = Conv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv2_1x1')(gap2)
        x3 = Activation('relu', name=f'{name}_relu2')(x3)
        x3 = Conv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv2_1x1_final')(x3)
        b = Activation('sigmoid', name=f'{name}_sigmoid_b')(x3)
        b2 = Conv2D(filters, (1, 1), padding='same', use_bias=True, name=f'{name}_conv_b2')(b)
        output = Add()([a2, b2])
        return output

# Helper function for Lightweight Feature Interaction Bottleneck (LFIB)
def lfib(inputs, filters, name, use_quantization=True):
    """Lightweight Feature Interaction Bottleneck (LFIB) - Paper Accurate"""
    
    # Initial 1x1 convolution
    if use_quantization:
        f1 = QConv2D(filters, 1, padding='same', name=f'{name}_f1_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        f1 = QActivation(qr, name=f'{name}_f1_act')(f1)
    else:
        f1 = Conv2D(filters, 1, padding='same', name=f'{name}_f1_conv')(inputs)
        f1 = Activation('relu', name=f'{name}_f1_act')(f1)

    # Channel split
    f21 = f1[:, :, :, :filters//2]
    f22 = f1[:, :, :, filters//2:]

    # CC modules
    CC1 = CC(f22, filters // 2, name_prefix=f'{name}_CC1')
    CC2 = CC(f21, filters // 2, name_prefix=f'{name}_CC2')

    # Left branch
    left_branch_input = Add(name=f'{name}_left_add')([f21, CC1])
    if use_quantization:
        left_out = QConv2D(filters // 2, (3, 1), padding='same', name=f'{name}_left_3x1', kernel_quantizer=kq, bias_quantizer=bq)(left_branch_input)
        left_out = QActivation(qr, name=f'{name}_left_3x1_act')(left_out)
        left_out = QConv2D(filters // 2, (1, 3), padding='same', name=f'{name}_left_1x3', kernel_quantizer=kq, bias_quantizer=bq)(left_out)
        left_out = QActivation(qr, name=f'{name}_left_1x3_act')(left_out)
    else:
        left_out = Conv2D(filters // 2, (3, 1), padding='same', name=f'{name}_left_3x1')(left_branch_input)
        left_out = Activation('relu', name=f'{name}_left_3x1_act')(left_out)
        left_out = Conv2D(filters // 2, (1, 3), padding='same', name=f'{name}_left_1x3')(left_out)
        left_out = Activation('relu', name=f'{name}_left_1x3_act')(left_out)

    # Right branch
    right_branch_input = Add(name=f'{name}_right_add')([f22, CC2])
    if use_quantization:
        right_out = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_right_dw', depthwise_quantizer=kq, bias_quantizer=bq)(right_branch_input)
        right_out = QActivation(qr, name=f'{name}_right_dw_act')(right_out)
        right_out = QConv2D(filters // 2, 1, padding='same', name=f'{name}_right_pw', kernel_quantizer=kq, bias_quantizer=bq)(right_out)
        right_out = QActivation(qr, name=f'{name}_right_pw_act')(right_out)
    else:
        right_out = DepthwiseConv2D((3, 3), padding='same', name=f'{name}_right_dw')(right_branch_input)
        right_out = Activation('relu', name=f'{name}_right_dw_act')(right_out)
        right_out = Conv2D(filters // 2, 1, padding='same', name=f'{name}_right_pw')(right_out)
        right_out = Activation('relu', name=f'{name}_right_pw_act')(right_out)

    # Fusion
    f2 = Concatenate(name=f'{name}_concat_branches')([left_out, right_out])
    cru_out = cru_block(f1, filters, f'{name}_cru', use_quantization=use_quantization)
    f2_plus_cru = Add(name=f'{name}_add_cru')([f2, cru_out])

    # Residual connection
    residual = Add(name=f'{name}_add_residual')([inputs, f2_plus_cru])

    # Final DW Conv + Channel Shuffle
    if use_quantization:
        final_out = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_final_dw', depthwise_quantizer=kq, bias_quantizer=bq)(residual)
        final_out = QActivation(qr, name=f'{name}_final_dw_act')(final_out)
    else:
        final_out = DepthwiseConv2D((3, 3), padding='same', name=f'{name}_final_dw')(residual)
        final_out = Activation('relu', name=f'{name}_final_dw_act')(final_out)
        
    shuffled_out = cgra4ml_channel_shuffle(final_out, groups=2, name=f'{name}_shuffle')
    
    return shuffled_out

def CC(inputs, filters, name_prefix):
    """Combination Coefficient Learning (CC) - CGRA4ML Compatible"""
    # Global Average Pooling
    gap = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]), name=f'{name_prefix}_gap_reshape')(gap)

    # Use Dense layers to learn channel-wise attention
    sdp = Dense(filters, activation='relu', name=f'{name_prefix}_sdp_dense1')(gap)
    sdp = Dense(filters, activation='sigmoid', name=f'{name_prefix}_sdp_dense2')(sdp)

    return Multiply(name=f'{name_prefix}_multiply')([inputs, sdp])

# Channel Reconstruction Unit (CRU)
def cru_block(x, filters, name, use_quantization=True):
    """CGRA4ML-Compatible Channel Reconstruction Unit (CRU)"""

    global_context = GlobalAveragePooling2D(name=f'{name}_gap')(x)
    global_context = Reshape((1, 1, x.shape[-1]))(global_context)
    if use_quantization:
        global_context = QConv2D(filters, 1, padding='same', activation='sigmoid', name=f'{name}_context_fc',
                                kernel_quantizer=kq, bias_quantizer=bq)(global_context)
    else:
        global_context = Conv2D(filters, 1, padding='same', activation='sigmoid', name=f'{name}_context_fc')(global_context)
    refined_x = Multiply()([x, global_context])
    return refined_x

# Improved Flatten Transformer (IFT)
def FLAM(inputs, num_heads=8, key_dim=64, window_size=7, dropout_rate=0.1, use_quantization=True):
    """CGRA4ML-Compatible Flattened Local Attention Module (FLAM)"""
    channels = inputs.shape[-1]
    if use_quantization:
        attn = QConv2D(filters=channels, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2), activation=qr, name="flam_attn_conv", kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        attn = Dropout(dropout_rate)(attn)
        attn = Add()([inputs, attn])
        attn = QBatchNormalization(name="flam_bn")(attn)
    else:
        attn = Conv2D(filters=channels, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2), activation='relu', name="flam_attn_conv")(inputs)
        attn = Dropout(dropout_rate)(attn)
        attn = Add()([inputs, attn])
        attn = BatchNormalization(renorm=True, name="flam_bn")(attn)
    out = attn
    if USE_DROPOUT_DURING_TRAINING:
        out = Dropout(0.1)(out)
    return out

def CAB(inputs, filters, use_quantization=True):
    """CGRA4ML-Compatible Context Aggregation Block (CAB)"""
    gap = GlobalAveragePooling2D()(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]))(gap)
    if use_quantization:
        mix = QConv2D(filters, 1, padding='same', activation=qr, name="cab_mix1", kernel_quantizer=kq, bias_quantizer=bq)(gap)
        mix = QConv2D(filters, 1, padding='same', activation='sigmoid', name="cab_mix2", kernel_quantizer=kq, bias_quantizer=bq)(mix)
    else:
        mix = Conv2D(filters, 1, padding='same', activation='relu', name="cab_mix1")(gap)
        mix = Conv2D(filters, 1, padding='same', activation='sigmoid', name="cab_mix2")(mix)
    out = Multiply()([inputs, mix])
    if USE_DROPOUT_DURING_TRAINING:
        out = Dropout(0.1)(out)
    return out


def improved_flatten_transformer(inputs, num_heads=8, key_dim=64, dropout_rate=0.1, use_quantization=True):
    """Improved Flatten Transformer: FLAM + CAB + CC"""
    filters = inputs.shape[-1]
    # FLAM: local/global attention
    x = FLAM(inputs, num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate, use_quantization=use_quantization)
    # CAB: context aggregation
    x = CAB(x, filters, use_quantization=use_quantization)
    # CC: channel-wise convolution
    x = CC(x, filters, name_prefix="ift_cc")
    return x

# Segmentation Head (SegHead)
def seghead(inputs, num_classes, use_quantization=True, config=MODEL_CONFIG_LMIA):
    """Segmentation Head with multi-scale fusion, following the paper's design."""
    # 1. Upsample and Concatenate features
    high_res_feature = inputs[-1]
    target_shape = high_res_feature.shape
    target_height, target_width = target_shape[1], target_shape[2]

    fused_features = []
    for feature in inputs:
        feature_shape = feature.shape
        if feature_shape[1] != target_height or feature_shape[2] != target_width:
            scale_height = target_height // feature_shape[1]
            scale_width = target_width // feature_shape[2]
            upsampled = UpSampling2D(size=(scale_height, scale_width), interpolation='nearest')(feature)
            fused_features.append(upsampled)
        else:
            fused_features.append(feature)
    
    x = Concatenate(axis=-1)(fused_features)

    # 2. Apply two convolution layers as per the paper
    seghead_filters = config.get('seghead_filters', 64) # Get from config or default to 64

    if use_quantization:
        # First 1x1 convolution
        x = QConv2D(seghead_filters, 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="seghead_conv1")(x)
        x = QActivation(qr, name="seghead_relu")(x)
        # Second 1x1 convolution (classifier)
        x = QConv2D(num_classes, 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="seghead_conv2")(x)
        output = Activation('softmax', name="output_softmax")(x)
    else:
        # First 1x1 convolution
        x = Conv2D(seghead_filters, 1, padding='same', name="seghead_conv1")(x)
        x = Activation('relu', name="seghead_relu")(x)
        # Second 1x1 convolution (classifier)
        x = Conv2D(num_classes, 1, padding='same', name="seghead_conv2")(x)
        output = Activation('softmax', name="output_softmax")(x)
        
    return output

# Encoder Block (initial block, downsampling, LFIB, FE)
def encoder(inputs, filters, use_quantization=True, name="enc"):
    """Encoder Block with LFIB and FE"""
    if use_quantization:
        x = QConv2D(filters, (3, 3), strides=2, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f'{name}_conv')(inputs)
        x = QActivation(qr, name=f'{name}_act')(x)
        if USE_DROPOUT_DURING_TRAINING:
            x = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(x)
    else:
        x = Conv2D(filters, (3, 3), strides=2, padding='same', name=f'{name}_conv')(inputs)
        x = Activation('relu', name=f'{name}_act')(x)
        if USE_DROPOUT_DURING_TRAINING:
            x = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(x)
    x = lfib(x, filters, f'{name}_lfib', use_quantization=use_quantization)
    x = feature_enhancement(x, filters, f'{name}_fe', use_quantization=use_quantization)
    return x

# Decoder Block (upsampling, LFIB, FE)
def decoder(inputs, skip_connection, filters, use_quantization=True, name="dec"):
    """Decoder Block with UpSampling, optional skip concat, and FE.

    Expects skip_connection to have same HxW as upsampled inputs. If None, skips concat.
    """
    # First upsample the decoder input
    x = UpSampling2D(size=(2, 2), interpolation='nearest', name=f'{name}_upsample')(inputs)

    # Concatenate with skip if provided
    if skip_connection is not None:
        x = Concatenate(axis=-1, name=f'{name}_concat')([x, skip_connection])

    x = feature_enhancement(x, filters, f'{name}_fe', use_quantization=use_quantization)

    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(x)
    return x

# LMIINet Model
def create_lmiinet(input_shape=(512, 1024, 3), num_classes=19, use_quantization=True, config=MODEL_CONFIG_LMIA):
    inputs = Input(shape=input_shape, name="input_image")

    # Initial downsampling stem
    if use_quantization:
        stem = QConv2D(config['encoder_filters'][0], (3, 3), strides=2, padding='same', name='stem_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        stem = QActivation(qr, name='stem_act')(stem)
    else:
        stem = Conv2D(config['encoder_filters'][0], (3, 3), strides=2, padding='same', name='stem_conv')(inputs)
        stem = Activation('relu', name='stem_act')(stem)

    # Encoder
    enc = stem
    encoder_outputs = [enc] # Start with stem output
    for i, f in enumerate(config['encoder_filters'][1:]): # Start from the second filter size
        enc = encoder(enc, f, use_quantization, name=f'enc{i+1}')
        encoder_outputs.append(enc)

    # Transformer
    transformer_output = improved_flatten_transformer(
        enc,
        num_heads=config.get('transformer_heads', 8),
        key_dim=config.get('transformer_key_dim', 64),
        dropout_rate=config.get('transformer_dropout', 0.1),
        use_quantization=use_quantization
    )

    # Decoder
    dec = transformer_output
    decoder_features = []
    for i, f in enumerate(config['decoder_filters']):
        # Align skip connections to the spatial size AFTER upsampling:
        # dec (1/16) -> upsample (1/8) should use encoder 1/8 (index -2), etc.
        skip_idx = -(i + 2)
        skip_connection = encoder_outputs[skip_idx] if abs(skip_idx) <= len(encoder_outputs) else None
        dec = decoder(dec, skip_connection, f, use_quantization, name=f'dec{i}')
        decoder_features.append(dec)

    # SegHead
    # The paper mentions using features from 1/2, 1/4, and 1/8 scales.
    # In our setup, this corresponds to the outputs of the last three decoder blocks.
    seg_output = seghead(decoder_features[-3:], num_classes, use_quantization, config=config)
    
    model = Model(inputs=inputs, outputs=seg_output, name="LMIINet_CGRA4ML_Paper_Accurate")
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
                        brightness_limit=0.6, contrast_limit=0.4, p=0.6
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
                # Standard Cityscapes label mapping (labelIds to trainIds) - follows official mapping
                # This maps the 34-class labelIds to 19-class trainIds as per Cityscapes standard
                CITYSCAPES_MAPPING = {
                    7: 0,   # road
                    8: 1,   # sidewalk  
                    11: 2,  # building
                    12: 3,  # wall
                    13: 4,  # fence
                    17: 5,  # pole
                    19: 6,  # traffic light
                    20: 7,  # traffic sign
                    21: 8,  # vegetation
                    22: 9,  # terrain
                    23: 10, # sky
                    24: 11, # person
                    25: 12, # rider
                    26: 13, # car
                    27: 14, # truck
                    28: 15, # bus
                    31: 16, # train
                    32: 17, # motorcycle
                    33: 18, # bicycle
                    # All other IDs map to ignore (255)
                }
                
                # Create mapped mask - default to ignore (255)
                mapped = np.full_like(mask, 255, dtype=np.uint8)
                for lid, tid in CITYSCAPES_MAPPING.items():
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

            # Ensure mask is int32 and ignore label 255 is kept as 255 (will be masked in loss)
            ignore_index = 255  # Use standard ignore index (255)
            mask = mask.astype(np.int32)
            # mask[mask == 255] already mapped to 255 above, keep as-is
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
        # Load sequentially; parallelism will come from model.fit(workers=...)
        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            if image is None or mask is None:
                continue
            # Main mask: [INPUT_HEIGHT, INPUT_WIDTH]
            mask_main = cv2.resize(mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_masks.append(mask_main)

        return np.array(batch_images), np.array(batch_masks)

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for categorical segmentation with ignore index support.
    """
    # Flatten spatial dimensions
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES])

    # Mask out ignore pixels (255)
    mask = tf.not_equal(y_true_flat, 255)
    y_true_masked = tf.boolean_mask(y_true_flat, mask)
    y_pred_masked = tf.boolean_mask(y_pred_flat, mask)

    # One-hot encode the valid pixels only
    y_true_oh = tf.one_hot(tf.cast(y_true_masked, tf.int32), NUM_CLASSES)
    y_pred_masked = tf.cast(y_pred_masked, tf.float32)

    intersection = tf.reduce_sum(y_true_oh * y_pred_masked)
    union = tf.reduce_sum(y_true_oh) + tf.reduce_sum(y_pred_masked)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice

def mean_iou(y_true, y_pred):
    """
    Mean IoU metric for semantic segmentation - ignores class 255.
    """
    # Mask out ignore pixels (255)
    mask = tf.not_equal(y_true, 255)
    y_true_masked = tf.boolean_mask(y_true, mask)
    
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_masked = tf.boolean_mask(y_pred_argmax, mask)
    
    y_true_masked = tf.cast(y_true_masked, tf.int32)
    y_pred_masked = tf.cast(y_pred_masked, tf.int32)
    
    ious = []
    for class_id in range(NUM_CLASSES):
        true_mask = tf.equal(y_true_masked, class_id)
        pred_mask = tf.equal(y_pred_masked, class_id)
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
        
        # Only compute IoU if class is present in ground truth
        class_present = tf.reduce_sum(tf.cast(true_mask, tf.float32)) > 0
        iou = tf.where(tf.logical_and(union > 0, class_present), intersection / union, tf.constant(float('nan')))
        ious.append(iou)
    
    ious = tf.stack(ious)
    valid_ious = tf.boolean_mask(ious, ~tf.math.is_nan(ious))
    result = tf.reduce_mean(valid_ious)
    return result

def accuracy(y_true, y_pred):
    """
    Computes pixel-wise accuracy, ignoring pixels with label 255 (ignore class).
    Args:
        y_true: ground truth labels, shape (batch, height, width)
        y_pred: predicted logits or probabilities, shape (batch, height, width, num_classes)
    Returns:
        Scalar accuracy value (float32)
    """
    # If y_pred is logits, convert to predicted class
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int64)
    mask = tf.not_equal(y_true, 255)
    matches = tf.equal(y_true, y_pred)
    matches = tf.logical_and(matches, mask)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(mask, tf.float32)
    # Avoid division by zero
    acc = tf.reduce_sum(matches) / (tf.reduce_sum(mask) + 1e-6)
    return acc

class BatchNormFreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_bn):
        super().__init__()
        self.freeze_bn = freeze_bn

    def on_epoch_begin(self, epoch, logs=None):
        if self.freeze_bn:
            # Unfreeze every 5th epoch (epoch numbers are 0-based)
            unfreeze = ((epoch) % 10 == 0)
            for layer in self.model.layers:
                if 'batchnorm' in layer.name.lower() or 'bn' in layer.name.lower():
                    layer.trainable = not self.freeze_bn or unfreeze
            if unfreeze:
                print(f"Unfreezing BN layers for epoch {epoch+1}")
            else:
                print(f"Freezing BN layers for epoch {epoch+1}")

class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, save_dir, num_samples=4, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.save_frequency = save_frequency  # Save every N epochs
        os.makedirs(save_dir, exist_ok=True)
        
        # Palette for official 19 trainIds (0..18) following Cityscapes standard
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
                            # Use new output key for masks
                            if 'output_softmax' in batch_masks_dict:
                                batch_masks = batch_masks_dict['output_softmax']
                            else:
                                # fallback for legacy keys
                                batch_masks = next(iter(batch_masks_dict.values()))
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
        monitor='val_mean_iou',
        save_best_only=False,  # Save every epoch, not just best
        save_weights_only=True,  # Save weights only to avoid Keras format issues
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    
    # Learning rate reduction with polynomial decay (as mentioned in paper)
    def poly_decay(epoch):
        max_epochs = EPOCHS
        return LEARNING_RATE * (1 - epoch / max_epochs) ** 0.9
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(poly_decay, verbose=1)
    callbacks.append(lr_scheduler)
    
    # Alternative: reduce on plateau (backup)
    lr_reducer = ReduceLROnPlateau(
        monitor='val_mean_iou',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        min_delta=0.008,
        verbose=1
    )
    # callbacks.append(lr_reducer)  # Commented out since we're using polynomial decay
    
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
    
    # Add BN freeze callback if requested
    if FREEZE_BN:
        callbacks.append(BatchNormFreezeCallback(FREEZE_BN))
        
    return callbacks

def load_compatible_weights(model, weights_path, skip_mismatched=True):
    """
    Load weights from a model with flexible layer matching.
    Handles cases where layer names or architectures have slightly changed.
    """
    print(f"Loading weights from: {weights_path}")
    
    try:
        # Try direct loading first
        model.load_weights(weights_path, by_name=True, skip_mismatch=skip_mismatched)
        print("Weight loading successful!")
        return True
        
    except Exception as e:
        print(f"Weight loading failed: {e}")
        print("This may be due to architecture differences.")
        return False

def inspect_model_weights(weights_path):
    """
    Inspect a weights file to understand its architecture and help with debugging.
    """
    print(f"Inspecting weights file: {weights_path}")
    
    try:
        # Try to load with current architecture
        temp_model = create_lmiinet(
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), 
            num_classes=NUM_CLASSES, 
            use_quantization=False, 
            config=MODEL_CONFIG_LMIA
        )
        temp_model.load_weights(weights_path)
        print(f"✅ Compatible with current {NUM_CLASSES}-class architecture")
        
        # Count layers
        total_layers = len(temp_model.layers)
        trainable_layers = len([l for l in temp_model.layers if len(l.get_weights()) > 0])
        
        print(f"   - Total layers: {total_layers}")
        print(f"   - Trainable layers: {trainable_layers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Not compatible with current architecture: {str(e)[:100]}...")
        return False

def convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG_LMIA):
    """Convert a trained float32 model to QAT model by replacing layers and transferring weights.
    This function is critical for CGRA4ML deployment as it ensures proper quantization-aware training.
    """
    print("Converting float32 model to QAT model for CGRA4ML...")
    
    # Create QAT model with quantization enabled (use_quantization=True for CGRA4ML)
    qat_model = create_lmiinet(
        input_shape=float32_model.input_shape[1:], 
        num_classes=config['num_classes'], 
        config=config, 
        use_quantization=True  # This enables CGRA4ML-compatible quantized layers
    )

    f_layers = {l.name: l for l in float32_model.layers}
    copied, skipped, qat_converted = 0, [], 0

    for ql in qat_model.layers:
        fl = f_layers.get(ql.name)
        if not fl or not hasattr(ql, 'get_weights') or not hasattr(fl, 'get_weights'):
            continue
        
        fw, qw = fl.get_weights(), ql.get_weights()

        try:
            # Handle Conv2D -> QConv2D conversion (CGRA4ML critical)
            if 'conv2d' in ql.name.lower() or 'qconv2d' in ql.name.lower():
                if len(fw) == len(qw) and all(f.shape == q.shape for f, q in zip(fw, qw)):
                    ql.set_weights(fw)
                    qat_converted += 1
                    continue
                elif len(fw) + 1 == len(qw):  # Float no-bias -> QAT with bias
                    ql.set_weights(fw + [np.zeros_like(qw[-1])])
                    qat_converted += 1
                    continue

            # Handle DepthwiseConv2D -> QDepthwiseConv2D conversion (CGRA4ML critical)
            elif 'depthwise' in ql.name.lower():
                if len(fw) == len(qw) and all(f.shape == q.shape for f, q in zip(fw, qw)):
                    ql.set_weights(fw)
                    qat_converted += 1
                    continue
                elif len(fw) + 1 == len(qw):  # Float no-bias -> QAT with bias
                    ql.set_weights(fw + [np.zeros_like(qw[-1])])
                    qat_converted += 1
                    continue

            # Handle BatchNormalization -> QBatchNormalization conversion (CGRA4ML critical)
            elif 'bn' in ql.name.lower() or 'batchnorm' in ql.name.lower():
                # BN renorm (float 7 tensors) -> QBN (4 tensors): take first 4 (gamma, beta, mean, var)
                if len(fw) > len(qw):
                    base = fw[:len(qw)]
                    if all(b.shape == q.shape for b, q in zip(base, qw)):
                        ql.set_weights(base)
                        qat_converted += 1
                        continue
                elif len(fw) == len(qw) and all(f.shape == q.shape for f, q in zip(fw, qw)):
                    ql.set_weights(fw)
                    qat_converted += 1
                    continue

            # Handle exact matches for other layers
            elif len(fw) == len(qw) and all(f.shape == q.shape for f, q in zip(fw, qw)):
                ql.set_weights(fw)
                copied += 1
                continue

            # Handle float no-bias -> QAT bias (append zeros)
            elif len(fw) + 1 == len(qw) and all(fw[i].shape == qw[i].shape for i in range(len(fw))):
                ql.set_weights(fw + [np.zeros_like(qw[-1])])
                qat_converted += 1
                continue

            # If we get here, weights couldn't be transferred
            skipped.append((ql.name, [w.shape for w in fw], [w.shape for w in qw]))
            
        except Exception as e:
            skipped.append((ql.name, str(e)))

    print(f"QAT Weight transfer: {copied} exact copies, {qat_converted} QAT conversions, {len(skipped)} skipped")
    if skipped[:3]:
        print("Examples of skipped layers:", skipped[:3])
    
    # Verify CGRA4ML compatibility
    qat_layers = sum(1 for l in qat_model.layers if 'Q' in type(l).__name__)
    total_layers = len([l for l in qat_model.layers if hasattr(l, 'get_weights') and l.get_weights()])
    print(f"CGRA4ML QAT model: {qat_layers}/{total_layers} layers quantized")
    
    return qat_model

def main():
    """Main training function"""
    print("=" * 60)
    print("CGRA4ML LMIINet Semantic Segmentation Training")
    print("=" * 60)
    print(f"Training with QAT: {TRAIN_QAT}")

    initial_epoch = 0  # Ensure initial_epoch is always defined

    # Initialize data generators
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

    # Model selection
    if not TRAIN_QAT:
        print("\n[PHASE 1] Baseline float32 training mode.")
        model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=False, config=MODEL_CONFIG_LMIA)
        if LOAD_MODEL_PATH:
            print(f"Attempting to load baseline model weights from: {LOAD_MODEL_PATH}")
            # Use improved weight loading with automatic conversion
            if load_compatible_weights(model, LOAD_MODEL_PATH, skip_mismatched=True):
                print("Baseline weights loaded successfully!")
            else:
                print("Warning: Could not load weights, starting with random initialization.")
            
            # Try to read epoch number from filename
            m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH)
            if m:
                initial_epoch = int(m.group(1))
                print(f"Resuming training from epoch {initial_epoch}.")

    # PHASE 2: QAT conversion and training (CGRA4ML-ready)
    elif TRAIN_QAT and not USE_AUTOKERAS:
        print("\n[PHASE 2] QAT conversion and training mode for CGRA4ML.")
        if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            if 'qat' in os.path.basename(LOAD_MODEL_PATH).lower():
                print(f"Detected QAT weights file: {LOAD_MODEL_PATH}")
                # Create QAT model with quantization enabled for CGRA4ML
                model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=True, config=MODEL_CONFIG_LMIA)
                try:
                    model.load_weights(LOAD_MODEL_PATH)
                    print("Loaded QAT weights successfully.")
                except Exception as e:
                    print(f"Error loading QAT weights: {e}")
            else:
                print(f"Detected baseline weights file: {LOAD_MODEL_PATH}")
                # Create baseline model and load weights with compatibility
                float32_model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=False, config=MODEL_CONFIG_LMIA)
                
                if load_compatible_weights(float32_model, LOAD_MODEL_PATH, skip_mismatched=True):
                    print("Loaded baseline weights successfully.")
                else:
                    print("Warning: Could not load baseline weights, converting from scratch.")
                
                print("Converting baseline to QAT model for CGRA4ML...")
                model = convert_float32_to_qat(float32_model, w_bits=8, a_bits=8, config=MODEL_CONFIG_LMIA)
                
                # Try to read epoch number from filename
                m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH or '')
                if m:
                    initial_epoch = int(m.group(1))
                    print(f"Resuming QAT from baseline epoch {initial_epoch} (for logging only).")
        else:
            print("Creating new QAT model (no weights found) for CGRA4ML.")
            # Create QAT model with quantization enabled for CGRA4ML
            model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=True, config=MODEL_CONFIG_LMIA)


    print("\nCompiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, ignore_class=255),
        # loss=dice_loss,
        metrics=[accuracy, dice_loss, mean_iou]
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

    # Save final model
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
