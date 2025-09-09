"""
LMIINet model architecture extraction from trainQAT_CGRA4ML.py
This module contains only the model creation functions without command line parsing
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from qkeras import *
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

def improved_flatten_transformer(x, config, use_quantization=True, name="transformer"):
    """Enhanced transformer block with proper scaling for flat features"""
    with tf.name_scope(name):
        # Get the input shape for calculations
        _, h, w, c = x.shape
        seq_len = h * w
        
        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        x_flat = tf.reshape(x, [-1, seq_len, c])
        
        # Multi-head self-attention with proper scaling
        mha = MultiHeadAttention(
            num_heads=config['transformer_heads'],
            key_dim=config['transformer_key_dim'],
            name=f"{name}_mha"
        )
        attn_output = mha(x_flat, x_flat)
        
        # Residual connection and layer norm
        x_flat = x_flat + attn_output
        x_flat = LayerNormalization(name=f"{name}_ln1")(x_flat)
        
        # Feed-forward network
        if use_quantization:
            ff = QDense(c, kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_ff1")(x_flat)
            ff = QActivation(qr, name=f"{name}_ff1_act")(ff)
            ff = QDense(c, kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_ff2")(ff)
        else:
            ff = Dense(c, name=f"{name}_ff1")(x_flat)
            ff = Activation('relu', name=f"{name}_ff1_act")(ff)
            ff = Dense(c, name=f"{name}_ff2")(ff)
        
        # Second residual connection and layer norm
        x_flat = x_flat + ff
        x_flat = LayerNormalization(name=f"{name}_ln2")(x_flat)
        
        # Reshape back to spatial dimensions: (B, H*W, C) -> (B, H, W, C)
        output = tf.reshape(x_flat, [-1, h, w, c])
        
        return output

def encoder(x, filters, use_quantization=True, name="encoder"):
    """Enhanced encoder block with skip connections"""
    with tf.name_scope(name):
        # Main path
        if use_quantization:
            conv1 = QConv2D(filters, 3, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_conv1")(x)
            conv1 = QActivation(qr, name=f"{name}_act1")(conv1)
            conv2 = QConv2D(filters, 3, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_conv2")(conv1)
            conv2 = QActivation(qr, name=f"{name}_act2")(conv2)
        else:
            conv1 = Conv2D(filters, 3, padding='same', name=f"{name}_conv1")(x)
            conv1 = Activation('relu', name=f"{name}_act1")(conv1)
            conv2 = Conv2D(filters, 3, padding='same', name=f"{name}_conv2")(conv1)
            conv2 = Activation('relu', name=f"{name}_act2")(conv2)
        
        # Skip connection if dimensions match
        if x.shape[-1] == filters:
            skip = x
        else:
            if use_quantization:
                skip = QConv2D(filters, 1, kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_skip")(x)
            else:
                skip = Conv2D(filters, 1, name=f"{name}_skip")(x)
        
        # Combine skip and main path
        combined = Add(name=f"{name}_add")([conv2, skip])
        
        # Downsample
        if use_quantization:
            output = QConv2D(filters, 3, strides=2, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_down")(combined)
            output = QActivation(qr, name=f"{name}_down_act")(output)
        else:
            output = Conv2D(filters, 3, strides=2, padding='same', name=f"{name}_down")(combined)
            output = Activation('relu', name=f"{name}_down_act")(output)
        
        return output

def decoder(x, skip, filters, use_quantization=True, name="decoder"):
    """Enhanced decoder block with skip connections"""
    with tf.name_scope(name):
        # Upsample
        if use_quantization:
            up = QConv2DTranspose(filters, 3, strides=2, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_up")(x)
            up = QActivation(qr, name=f"{name}_up_act")(up)
        else:
            up = Conv2DTranspose(filters, 3, strides=2, padding='same', name=f"{name}_up")(x)
            up = Activation('relu', name=f"{name}_up_act")(up)
        
        # Concatenate with skip connection
        concat = Concatenate(name=f"{name}_concat")([up, skip])
        
        # Refine features
        if use_quantization:
            conv1 = QConv2D(filters, 3, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_conv1")(concat)
            conv1 = QActivation(qr, name=f"{name}_act1")(conv1)
            conv2 = QConv2D(filters, 3, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_conv2")(conv1)
            conv2 = QActivation(qr, name=f"{name}_act2")(conv2)
        else:
            conv1 = Conv2D(filters, 3, padding='same', name=f"{name}_conv1")(concat)
            conv1 = Activation('relu', name=f"{name}_act1")(conv1)
            conv2 = Conv2D(filters, 3, padding='same', name=f"{name}_conv2")(conv1)
            conv2 = Activation('relu', name=f"{name}_act2")(conv2)
        
        return conv2

def segmentation_head(x, num_classes, use_quantization=True, name="seg_head"):
    """Segmentation head for final output"""
    with tf.name_scope(name):
        if use_quantization:
            conv = QConv2D(64, 3, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f"{name}_conv")(x)
            conv = QActivation(qr, name=f"{name}_act")(conv)
            # Final layer without quantization for better accuracy
            output = Conv2D(num_classes, 1, padding='same', name=f"{name}_output")(conv)
        else:
            conv = Conv2D(64, 3, padding='same', name=f"{name}_conv")(x)
            conv = Activation('relu', name=f"{name}_act")(conv)
            output = Conv2D(num_classes, 1, padding='same', name=f"{name}_output")(conv)
        
        return output

def create_lmiinet(input_shape=(512, 1024, 3), num_classes=19, use_quantization=True, config=MODEL_CONFIG_LMIA, aux_out=True, inference_mode=False):
    """Create LMIINet model with the exact architecture from trainQAT_CGRA4ML.py"""
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
        config,
        use_quantization,
        name='transformer'
    )

    # Decoder with skip connections
    dec = transformer_output
    for i, f in enumerate(config['decoder_filters']):
        skip_idx = len(encoder_outputs) - 2 - i  # Get corresponding encoder output
        if skip_idx >= 0:
            skip = encoder_outputs[skip_idx]
            dec = decoder(dec, skip, f, use_quantization, name=f'dec{i+1}')
        else:
            # No more skip connections, just upsample
            if use_quantization:
                dec = QConv2DTranspose(f, 3, strides=2, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name=f'dec{i+1}_up')(dec)
                dec = QActivation(qr, name=f'dec{i+1}_up_act')(dec)
            else:
                dec = Conv2DTranspose(f, 3, strides=2, padding='same', name=f'dec{i+1}_up')(dec)
                dec = Activation('relu', name=f'dec{i+1}_up_act')(dec)

    # Final segmentation head
    main_output = segmentation_head(dec, num_classes, use_quantization, name='main_seg')
    
    # Create model
    if aux_out and not inference_mode:
        # Auxiliary output from transformer for training
        aux_output = segmentation_head(
            UpSampling2D(size=(8, 8), name='aux_upsample')(transformer_output),
            num_classes, 
            use_quantization, 
            name='aux_seg'
        )
        model = keras.Model(inputs=inputs, outputs=[main_output, aux_output], name="LMIINet")
    else:
        model = keras.Model(inputs=inputs, outputs=main_output, name="LMIINet")
    
    return model
