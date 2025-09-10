"""
LMIINet model architecture for CGRA4ML - QAT version only
This module contains only the QAT model creation functions without dropout
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, BatchNormalization, Activation, 
    GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Add, 
    Multiply, Concatenate, UpSampling2D
)
from tensorflow.keras.models import Model

from qkeras import (
    QConv2D, QDepthwiseConv2D, QBatchNormalization, QActivation, QDense,
    quantized_bits, quantized_relu
)
import os

# Model configuration
MODEL_CONFIG_LMIA = {
    'encoder_filters': [24, 48, 96, 128],   # go from 24 to 128
    'decoder_filters': [128, 96, 48, 24],     # mirror down
    'transformer_heads': 8,                 # fewer heads
    'transformer_key_dim': 64,              # smaller projection
    'transformer_dropout': 0.1,
    'seghead_filters': 64,                   # Same
    'num_classes': 19                        # Correct Cityscapes classes
}

# Quantizer helpers for QKeras
X_BITS = 8      # activations (hw.bits_input)
K_BITS = 8      # weights     (hw.bits_weights)
B_BITS = 8      # bias        (hw.bits_bias)
QINT_BITS = 0   # integer bits, same as example

kq = f"quantized_bits({K_BITS},{QINT_BITS},False,True,1)"          # weights
bq = f"quantized_bits({B_BITS},{QINT_BITS},False,True,1)"          # bias
qr = f"quantized_relu({X_BITS},{QINT_BITS},negative_slope=0)"      # ReLU
qb = f"quantized_bits({X_BITS},{QINT_BITS},False,False,1)"         # requant (identity/clip)

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
def feature_enhancement(inputs, filters, name):
    """CGRA4ML-Compatible Feature Enhancement Block - QAT version only"""
    
    # t.Mean: Apply Global Average Pooling (GAP)
    gap = GlobalAveragePooling2D()(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]))(gap)

    # t.Max: Apply Global Max Pooling (GMP)
    gmp = GlobalMaxPooling2D()(inputs)
    gmp = Reshape((1, 1, inputs.shape[-1]))(gmp)

    # Concatenate GAP and GMP along the channel axis (axis=-1)
    x = Concatenate(axis=-1)([gap, gmp])

    # QAT branch only
    x = QConv2D(filters, (3, 3), padding='same', use_bias=True, name=f'{name}_conv_3x3', kernel_quantizer=kq, bias_quantizer=bq)(x)
    x = QBatchNormalization(name=f'{name}_bn')(x)
    a = QActivation('sigmoid', name=f'{name}_sigmoid')(x)

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

# Helper function for Lightweight Feature Interaction Bottleneck (LFIB)
def lfib(inputs, filters, name):
    """Lightweight Feature Interaction Bottleneck (LFIB) - Paper Accurate"""

    f1 = QConv2D(filters, 1, padding='same', name=f'{name}_f1_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
    f1 = QActivation(qr, name=f'{name}_f1_act')(f1)

    # Channel split
    f21 = f1[:, :, :, :filters//2]
    f22 = f1[:, :, :, filters//2:]

    # CC modules
    CC1 = CC(f22, filters // 2, name_prefix=f'{name}_CC1')
    CC2 = CC(f21, filters // 2, name_prefix=f'{name}_CC2')

    # Left branch
    left_branch_input = Add(name=f'{name}_left_add')([f21, CC1])

    left_out = QConv2D(filters // 2, (3, 1), padding='same', name=f'{name}_left_3x1', kernel_quantizer=kq, bias_quantizer=bq)(left_branch_input)
    left_out = QActivation(qr, name=f'{name}_left_3x1_act')(left_out)
    left_out = QConv2D(filters // 2, (1, 3), padding='same', name=f'{name}_left_1x3', kernel_quantizer=kq, bias_quantizer=bq)(left_out)
    left_out = QActivation(qr, name=f'{name}_left_1x3_act')(left_out)


    # Right branch
    right_branch_input = Add(name=f'{name}_right_add')([f22, CC2])

    right_out = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_right_dw', depthwise_quantizer=kq, bias_quantizer=bq)(right_branch_input)
    right_out = QActivation(qr, name=f'{name}_right_dw_act')(right_out)
    right_out = QConv2D(filters // 2, 1, padding='same', name=f'{name}_right_pw', kernel_quantizer=kq, bias_quantizer=bq)(right_out)
    right_out = QActivation(qr, name=f'{name}_right_pw_act')(right_out)


    # Fusion
    f2 = Concatenate(name=f'{name}_concat_branches')([left_out, right_out])
    cru_out = cru_block(f1, filters, f'{name}_cru')
    f2_plus_cru = Add(name=f'{name}_add_cru')([f2, cru_out])

    # Residual connection
    residual = Add(name=f'{name}_add_residual')([inputs, f2_plus_cru])

    # Final DW Conv + Channel Shuffle
    final_out = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_final_dw', depthwise_quantizer=kq, bias_quantizer=bq)(residual)
    final_out = QActivation(qr, name=f'{name}_final_dw_act')(final_out)

        
    shuffled_out = cgra4ml_channel_shuffle(final_out, groups=2, name=f'{name}_shuffle')
    
    return shuffled_out

def CC(inputs, filters, name_prefix):
    """Combination Coefficient Learning (CC) - CGRA4ML Compatible - QAT version only"""
    # Global Average Pooling
    gap = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]), name=f'{name_prefix}_gap_reshape')(gap)

    # Use Dense layers to learn channel-wise attention
    sdp = Dense(filters, activation='relu', name=f'{name_prefix}_sdp_dense1')(gap)
    sdp = Dense(filters, activation='sigmoid', name=f'{name_prefix}_sdp_dense2')(sdp)

    return Multiply(name=f'{name_prefix}_multiply')([inputs, sdp])

# Channel Reconstruction Unit (CRU)
def cru_block(x, filters, name):
    """CGRA4ML-Compatible Channel Reconstruction Unit (CRU) - QAT version only"""

    global_context = GlobalAveragePooling2D(name=f'{name}_gap')(x)
    global_context = Reshape((1, 1, x.shape[-1]))(global_context)
    global_context = QConv2D(filters, 1, padding='same', activation='sigmoid', name=f'{name}_context_fc',
                                kernel_quantizer=kq, bias_quantizer=bq)(global_context)
    refined_x = Multiply()([x, global_context])
    return refined_x

# Improved Flatten Transformer (IFT)
def FLAM(inputs, num_heads=8, key_dim=64, window_size=7, dropout_rate=0.1):
    """CGRA4ML-Compatible Flattened Local Attention Module (FLAM) - QAT version only"""
    channels = inputs.shape[-1]
    attn = QConv2D(filters=channels, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2), activation=qr, name="flam_attn_conv", kernel_quantizer=kq, bias_quantizer=bq)(inputs)
    attn = Add()([inputs, attn])
    attn = QBatchNormalization(name="flam_bn")(attn)

    out = attn
    return out

def CAB(inputs, filters):
    """CGRA4ML-Compatible Context Aggregation Block (CAB) - QAT version only"""
    gap = GlobalAveragePooling2D()(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]))(gap)
    # QAT version only
    mix = QConv2D(filters, 1, padding='same', activation=qr, name="cab_mix1", kernel_quantizer=kq, bias_quantizer=bq)(gap)
    mix = QConv2D(filters, 1, padding='same', activation='sigmoid', name="cab_mix2", kernel_quantizer=kq, bias_quantizer=bq)(mix)
    out = Multiply()([inputs, mix])
    return out


def improved_flatten_transformer(inputs, num_heads=8, key_dim=64, dropout_rate=0.1):
    """Improved Flatten Transformer: FLAM + CAB + CC"""
    filters = inputs.shape[-1]
    # FLAM: local/global attention
    x = FLAM(inputs, num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate)
    # CAB: context aggregation
    x = CAB(x, filters, )
    # CC: channel-wise convolution
    x = CC(x, filters, name_prefix="ift_cc")
    return x

# Segmentation Head (SegHead)
def seghead(inputs, num_classes, config=MODEL_CONFIG_LMIA):
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
    
    # Final 1x1 convolution to map concatenated features to num_classes

    x = QConv2D(num_classes, 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="seghead_final_conv")(x)
    output = Activation('softmax', name="output_softmax")(x)

    return output

# Encoder Block (initial block, downsampling, LFIB, FE)
def encoder(inputs, filters, name="enc"):
    """Encoder Block with LFIB and FE - QAT version only"""
    # QAT version only
    x = QConv2D(filters, (3, 3), strides=2, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f'{name}_conv')(inputs)
    x = QActivation(qr, name=f'{name}_act')(x)
    x = lfib(x, filters, f'{name}_lfib')
    x = feature_enhancement(x, filters, f'{name}_fe')
    return x

# Decoder Block (upsampling, LFIB, FE)
def decoder(inputs, skip_connection, filters, name="dec"):
    """Decoder Block with UpSampling, optional skip concat, and FE - QAT version only.

    Expects skip_connection to have same HxW as upsampled inputs. If None, skips concat.
    """
    # First upsample the decoder input
    x = UpSampling2D(size=(2, 2), interpolation='nearest', name=f'{name}_upsample')(inputs)

    # Concatenate with skip if provided
    if skip_connection is not None:
        x = Concatenate(axis=-1, name=f'{name}_concat')([x, skip_connection])

    x = feature_enhancement(x, filters, f'{name}_fe')
    # Spatial refinement (DW+PW) to smooth blocky upsampling artifacts - QAT version only
    x = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_refine_dw', depthwise_quantizer=kq, bias_quantizer=bq)(x)
    x = QActivation(qr, name=f'{name}_refine_dw_act')(x)
    x = QConv2D(filters, 1, padding='same', name=f'{name}_refine_pw', kernel_quantizer=kq, bias_quantizer=bq)(x)
    x = QActivation(qr, name=f'{name}_refine_pw_act')(x)

    return x

# LMIINet Model
def create_lmiinet(input_shape=(512, 1024, 3), num_classes=19, config=MODEL_CONFIG_LMIA, aux_out=True, inference_mode=False):
    inputs = Input(shape=input_shape, name="input_image")

    # Initial downsampling stem
    stem = QConv2D(config['encoder_filters'][0], (3, 3), strides=2, padding='same', name='stem_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
    stem = QActivation(qr, name='stem_act')(stem)


    # Encoder
    enc = stem
    encoder_outputs = [enc] # Start with stem output
    for i, f in enumerate(config['encoder_filters'][1:]): # Start from the second filter size
        enc = encoder(enc, f, name=f'enc{i+1}')
        encoder_outputs.append(enc)

    # Transformer
    transformer_output = improved_flatten_transformer(
        enc,
        num_heads=config.get('transformer_heads', 8),
        key_dim=config.get('transformer_key_dim', 64),
        dropout_rate=config.get('transformer_dropout', 0.1)
    )

    # Decoder
    dec = transformer_output
    decoder_features = []
    for i, f in enumerate(config['decoder_filters']):
        # Align skip connections to the spatial size AFTER upsampling:
        # dec (1/16) -> upsample (1/8) should use encoder 1/8 (index -2), etc.
        skip_idx = -(i + 2)
        skip_connection = encoder_outputs[skip_idx] if abs(skip_idx) <= len(encoder_outputs) else None
        dec = decoder(dec, skip_connection, f, name=f'dec{i}')
        decoder_features.append(dec)

    # Main SegHead
    seg_output = seghead(decoder_features[-3:], num_classes, config=config)
    
    # Auxiliary segmentation head applied to encoder output (enc)
        # Single output model (no auxiliary loss)
    model = Model(inputs=inputs, outputs=seg_output, name="LMIINet_CGRA4ML_Paper_Accurate")
    
    return model