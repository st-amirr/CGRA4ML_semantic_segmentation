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
import cv2
import threading
import time
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Dropout, Dense, 
    Add, DepthwiseConv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, 
    Reshape, Multiply, MultiHeadAttention, GlobalMaxPooling2D
)
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu, QDense, QDepthwiseConv2D
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.metrics import MeanIoU
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from PIL import Image
from pathlib import Path
import math
import matplotlib.pyplot as plt
import keras_tuner as kt
from matplotlib.colors import ListedColormap
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)


epochs = 140
batch_size = 3
LOAD_MODEL_PATH = None
SUBSET_RATIO =  0.9

AUGMENTATION = True
TRAIN_QAT = True
LEARNING_RATE = 7e-4
FREEZE_DEC = False
FREEZE_BN = True 
USE_DROPOUT_DURING_TRAINING = True

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 19
NUM_WORKERS = 6
DEBUG_MODE = False


MODEL_CONFIG_LMIA = {
    'encoder_filters': [24, 48, 96, 128],
    'decoder_filters': [128, 96, 48, 24],
    'transformer_heads': 8,
    'transformer_key_dim': 64,
    'transformer_dropout': 0.1,
    'seghead_filters': 64,
    'num_classes': 19
}

TRAIN_IMAGES_DIR = "/data/leftImg8bit/train"
TRAIN_MASKS_DIR = "/data/gtFine/train"
VAL_IMAGES_DIR = "/data/leftImg8bit/val"
VAL_MASKS_DIR = "/data/gtFine/val"

MODELS_DIR = "/workspace/models"
TB_LOGS_DIR = "/workspace/models/tb_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TB_LOGS_DIR, exist_ok=True)
X_BITS = 8
K_BITS = 8
B_BITS = 8
QINT_BITS = 0

kq = f"quantized_bits({K_BITS},{QINT_BITS},False,True,1)"
bq = f"quantized_bits({B_BITS},{QINT_BITS},False,True,1)"
qr = f"quantized_relu({X_BITS},{QINT_BITS},negative_slope=0)"
qb = f"quantized_bits({X_BITS},{QINT_BITS},False,False,1)"


n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(n_cores // 2)
tf.config.set_soft_device_placement(True)


gpu = tf.config.experimental.list_physical_devices("GPU")

tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)

tf.debugging.set_log_device_placement(debug_mode)




class ZoomToSignsAndLights(DualTransform):

    def __init__(
        self,
        target_classes=[4, 5, 6, 7, 11],
        zoom_scale=2.0,
        min_object_size=5,
        target_size=(1024, 512),
        p=0.20,
        always_apply=False,
    ):
        super(ZoomToSignsAndLights, self).__init__(always_apply, p)
        self.target_classes = target_classes
        self.zoom_scale = zoom_scale
        self.min_object_size = min_object_size
        self.target_size = target_size
    
    def apply(self, img, center_x=0, center_y=0, **params):
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        
        crop_w = int(target_w * self.zoom_scale)
        crop_h = int(target_h * self.zoom_scale)
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        cropped = img[y1:y2, x1:x2]
        
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def apply_to_mask(self, mask, center_x=0, center_y=0, **params):
        h, w = mask.shape[:2]
        target_w, target_h = self.target_size
        
        crop_w = int(target_w * self.zoom_scale)
        crop_h = int(target_h * self.zoom_scale)
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        cropped = mask[y1:y2, x1:x2]
        
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        return resized
    
    def get_params(self):
        return {
            "center_x": 0,
            "center_y": 0,
        }
    
    def apply_detection(self, img, mask, **params):

        target_objects = []
        
        for class_id in self.target_classes:
            class_mask = (mask == class_id)
            
            if np.any(class_mask):
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    class_mask.astype(np.uint8), connectivity=8
                )
                
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= self.min_object_size:
                        center_x = int(centroids[i][0])
                        center_y = int(centroids[i][1])
                        target_objects.append((center_x, center_y, area))
        
        if not target_objects:
            h, w = img.shape[:2]
            target_w, target_h = self.target_size
            
            x1 = (w - target_w) // 2
            y1 = (h - target_h) // 2
            x2 = x1 + target_w
            y2 = y1 + target_h
            
            img_cropped = img[y1:y2, x1:x2]
            mask_cropped = mask[y1:y2, x1:x2]
            
            return img_cropped, mask_cropped
        
        center_x, center_y, area = target_objects[np.random.randint(len(target_objects))]
        
        img_zoomed = self.apply(img, center_x=center_x, center_y=center_y)
        
        mask_zoomed = self.apply_to_mask(mask, center_x=center_x, center_y=center_y)
        
        return img_zoomed, mask_zoomed
    
    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            raise NotImplementedError("Replay mode is not supported for this transform")
        
        if force_apply or self.apply_with_params(self.params, **kwargs):
            if 'image' in kwargs and 'mask' in kwargs:
                kwargs['image'], kwargs['mask'] = self.apply_detection(kwargs['image'], kwargs['mask'])
            else:
                img = kwargs['image']
                h, w = img.shape[:2]
                target_w, target_h = self.target_size
                
                x1 = (w - target_w) // 2
                y1 = (h - target_h) // 2
                x2 = x1 + target_w
                y2 = y1 + target_h
                
                kwargs['image'] = img[y1:y2, x1:x2]
        
        return kwargs



def cgra4ml_channel_shuffle(inputs, groups=2, name="shuffle"):
   
    group_1 = inputs[:, :, :, :inputs.shape[-1] // groups]
    group_2 = inputs[:, :, :, inputs.shape[-1] // groups:]
    
    shuffled = Concatenate(axis=-1, name=f'{name}_concat')([group_2, group_1])
    
    return shuffled

def feature_enhancement(inputs, filters, name, use_quantization=True):
    
    gap = GlobalAveragePooling2D()(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]))(gap)

    gmp = GlobalMaxPooling2D()(inputs)
    gmp = Reshape((1, 1, inputs.shape[-1]))(gmp)

    x = Concatenate(axis=-1)([gap, gmp])

    if use_quantization:
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

def lfib(inputs, filters, name, use_quantization=True):
    
    if use_quantization:
        f1 = QConv2D(filters, 1, padding='same', name=f'{name}_f1_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        f1 = QActivation(qr, name=f'{name}_f1_act')(f1)
    else:
        f1 = Conv2D(filters, 1, padding='same', name=f'{name}_f1_conv')(inputs)
        f1 = Activation('relu', name=f'{name}_f1_act')(f1)

    f21 = f1[:, :, :, :filters//2]
    f22 = f1[:, :, :, filters//2:]

    CC1 = CC(f22, filters // 2, name_prefix=f'{name}_CC1')
    CC2 = CC(f21, filters // 2, name_prefix=f'{name}_CC2')

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

    f2 = Concatenate(name=f'{name}_concat_branches')([left_out, right_out])
    cru_out = cru_block(f1, filters, f'{name}_cru', use_quantization=use_quantization)
    f2_plus_cru = Add(name=f'{name}_add_cru')([f2, cru_out])

    residual = Add(name=f'{name}_add_residual')([inputs, f2_plus_cru])

    if use_quantization:
        final_out = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_final_dw', depthwise_quantizer=kq, bias_quantizer=bq)(residual)
        final_out = QActivation(qr, name=f'{name}_final_dw_act')(final_out)
    else:
        final_out = DepthwiseConv2D((3, 3), padding='same', name=f'{name}_final_dw')(residual)
        final_out = Activation('relu', name=f'{name}_final_dw_act')(final_out)
        
    shuffled_out = cgra4ml_channel_shuffle(final_out, groups=2, name=f'{name}_shuffle')
    
    return shuffled_out

def CC(inputs, filters, name_prefix):
    gap = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(inputs)
    gap = Reshape((1, 1, inputs.shape[-1]), name=f'{name_prefix}_gap_reshape')(gap)

    sdp = Dense(filters, activation='relu', name=f'{name_prefix}_sdp_dense1')(gap)
    sdp = Dense(filters, activation='sigmoid', name=f'{name_prefix}_sdp_dense2')(sdp)

    return Multiply(name=f'{name_prefix}_multiply')([inputs, sdp])

def cru_block(x, filters, name, use_quantization=True):

    global_context = GlobalAveragePooling2D(name=f'{name}_gap')(x)
    global_context = Reshape((1, 1, x.shape[-1]))(global_context)
    if use_quantization:
        global_context = QConv2D(filters, 1, padding='same', activation='sigmoid', name=f'{name}_context_fc',
                                kernel_quantizer=kq, bias_quantizer=bq)(global_context)
    else:
        global_context = Conv2D(filters, 1, padding='same', activation='sigmoid', name=f'{name}_context_fc')(global_context)
    refined_x = Multiply()([x, global_context])
    return refined_x

def FLAM(inputs, num_heads=8, key_dim=64, window_size=7, dropout_rate=0.1, use_quantization=True):
    channels = inputs.shape[-1]
    if use_quantization:
        attn = QConv2D(filters=channels, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2), activation=qr, name="flam_attn_conv", kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        attn = Add()([inputs, attn])
        attn = QBatchNormalization(name="flam_bn")(attn)
    else:
        attn = Conv2D(filters=channels, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2), activation='relu', name="flam_attn_conv")(inputs)
        attn = Add()([inputs, attn])
        attn = BatchNormalization(renorm=True, name="flam_bn")(attn)
    out = attn
    if USE_DROPOUT_DURING_TRAINING:
        out = Dropout(0.1)(out)
    return out

def CAB(inputs, filters, use_quantization=True):
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
    filters = inputs.shape[-1]
    x = FLAM(inputs, num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate, use_quantization=use_quantization)
    x = CAB(x, filters, use_quantization=use_quantization)
    x = CC(x, filters, name_prefix="ift_cc")
    return x

def seghead(inputs, num_classes, use_quantization=True, config=MODEL_CONFIG_LMIA):
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
    
    if use_quantization:
        x = QConv2D(num_classes, 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="seghead_final_conv")(x)
        output = Activation('softmax', name="output_softmax")(x)
    else:
        x = Conv2D(num_classes, 1, padding='same', name="seghead_final_conv")(x)
        output = Activation('softmax', name="output_softmax")(x)
    
    return output

def encoder(inputs, filters, use_quantization=True, name="enc"):
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

def decoder(inputs, skip_connection, filters, use_quantization=True, name="dec"):
    x = UpSampling2D(size=(2, 2), interpolation='nearest', name=f'{name}_upsample')(inputs)

    if skip_connection is not None:
        x = Concatenate(axis=-1, name=f'{name}_concat')([x, skip_connection])

    x = feature_enhancement(x, filters, f'{name}_fe', use_quantization=use_quantization)
    if use_quantization:
        x = QDepthwiseConv2D((3, 3), padding='same', name=f'{name}_refine_dw', depthwise_quantizer=kq, bias_quantizer=bq)(x)
        x = QActivation(qr, name=f'{name}_refine_dw_act')(x)
        x = QConv2D(filters, 1, padding='same', name=f'{name}_refine_pw', kernel_quantizer=kq, bias_quantizer=bq)(x)
        x = QActivation(qr, name=f'{name}_refine_pw_act')(x)
    else:
        x = DepthwiseConv2D((3, 3), padding='same', name=f'{name}_refine_dw')(x)
        x = Activation('relu', name=f'{name}_refine_dw_act')(x)
        x = Conv2D(filters, 1, padding='same', name=f'{name}_refine_pw')(x)
        x = Activation('relu', name=f'{name}_refine_pw_act')(x)

    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(MODEL_CONFIG_LMIA['transformer_dropout'])(x)
    return x

def create_lmiinet(input_shape=(512, 1024, 3), num_classes=19, use_quantization=True, config=MODEL_CONFIG_LMIA, aux_out=True, inference_mode=False):
    inputs = Input(shape=input_shape, name="input_image")

    if use_quantization:
        stem = QConv2D(config['encoder_filters'][0], (3, 3), strides=2, padding='same', name='stem_conv', kernel_quantizer=kq, bias_quantizer=bq)(inputs)
        stem = QActivation(qr, name='stem_act')(stem)
    else:
        stem = Conv2D(config['encoder_filters'][0], (3, 3), strides=2, padding='same', name='stem_conv')(inputs)
        stem = Activation('relu', name='stem_act')(stem)

    enc = stem
    encoder_outputs = [enc]
    for i, f in enumerate(config['encoder_filters'][1:]):
        enc = encoder(enc, f, use_quantization, name=f'enc{i+1}')
        encoder_outputs.append(enc)

    transformer_output = improved_flatten_transformer(
        enc,
        num_heads=config.get('transformer_heads', 8),
        key_dim=config.get('transformer_key_dim', 64),
        dropout_rate=config.get('transformer_dropout', 0.1),
        use_quantization=use_quantization
    )

    dec = transformer_output
    decoder_features = []
    for i, f in enumerate(config['decoder_filters']):
        skip_idx = -(i + 2)
        skip_connection = encoder_outputs[skip_idx] if abs(skip_idx) <= len(encoder_outputs) else None
        dec = decoder(dec, skip_connection, f, use_quantization, name=f'dec{i}')
        decoder_features.append(dec)

    seg_output = seghead(decoder_features[-3:], num_classes, use_quantization, config=config)
    
    if aux_out:
        if use_quantization:
            aux_conv = QConv2D(config.get('seghead_filters', 64), 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="aux_seghead_conv1")(enc)
            aux_conv = QActivation(qr, name="aux_seghead_relu")(aux_conv)
            aux_conv = QConv2D(num_classes, 1, padding='same', kernel_quantizer=kq, bias_quantizer=bq, name="aux_seghead_conv2")(aux_conv)
            aux_output = Activation('softmax', name="aux_output_softmax")(aux_conv)
        else:
            aux_conv = Conv2D(config.get('seghead_filters', 64), 1, padding='same', name="aux_seghead_conv1")(enc)
            aux_conv = Activation('relu', name="aux_seghead_relu")(aux_conv)
            aux_conv = Conv2D(num_classes, 1, padding='same', name="aux_seghead_conv2")(aux_conv)
            aux_output = Activation('softmax', name="aux_output_softmax")(aux_conv)
        
        model = Model(inputs=inputs, outputs=[seg_output, aux_output], name="LMIINet_CGRA4ML_Paper_Accurate_With_Aux")
    else:
        model = Model(inputs=inputs, outputs=seg_output, name="LMIINet_CGRA4ML_Paper_Accurate")
    
    return model


class CityscapesDataGenerator(tf.keras.utils.Sequence):

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
        super().__init__(**kwargs)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

        self.image_paths = list(self.image_dir.glob("**/*.png"))
        if subset_ratio < 1.0:
            subset_size = int(len(self.image_paths) * subset_ratio)
            self.image_paths = self.image_paths[:subset_size]


        if self.augment:
            self.transform = A.Compose(
                [   
                    ZoomToSignsAndLights(
                        target_classes=[4, 5, 6, 7, 11],
                        zoom_scale=2.0,
                        min_object_size=5,
                        target_size=(self.target_width, self.target_height),
                        p=0.20,
                    ),
                    A.RandomCrop(
                        width=self.target_width, height=self.target_height, p=0.1
                    ),
                    A.HorizontalFlip(p=0.3),
                    A.Rotate(limit=5, p=0.2, border_mode=cv2.BORDER_CONSTANT),
                    A.Affine(
                        scale=(0.8, 1.2),
                        rotate=(-15, 15),
                        translate_percent=(0.2, 0.2),
                        p=0.1
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.2, p=0.2
                    ),
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.2
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=30,
                        p=0.1,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.1),
                    A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0, 0), per_channel=False, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                    A.RandomRain(
                        drop_length=20,
                        drop_width=1,
                        drop_color=(200, 200, 200),
                        blur_value=3,
                        brightness_coefficient=0.7,
                        rain_type="drizzle",
                        p=0.05,
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.1
                    ),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(2, 16),
                        hole_width_range=(2, 16),
                        fill=0,
                        fill_mask=0,
                        p=0.1,
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
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.float32), np.zeros((self.target_height, self.target_width), dtype=np.int32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self._get_mask_path(image_path)
        if not mask_path.exists():
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.float32), np.zeros((self.target_height, self.target_width), dtype=np.int32)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.float32), np.zeros((self.target_height, self.target_width), dtype=np.int32)

        if "_labelTrainIds.png" in str(mask_path):
            pass
        else:
            CITYSCAPES_MAPPING = {
                7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18, 255: 255
            }
            
            mapped = np.full_like(mask, 255, dtype=np.uint8)
            for lid, tid in CITYSCAPES_MAPPING.items():
                mapped[mask == lid] = tid
            mask = mapped

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

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

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.int32)
        return image, mask

    def _get_mask_path(self, image_path):
        mask_name_tids = image_path.name.replace(
            "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
        )
        mask_path_tids = self.mask_dir / image_path.parent.name / mask_name_tids

        if mask_path_tids.exists():
            return mask_path_tids
        else:
            mask_name_lids = image_path.name.replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"
            )
            mask_path_lids = self.mask_dir / image_path.parent.name / mask_name_lids
            return mask_path_lids

    def __getitem__(self, index):
        batch_paths = self.image_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []
        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            if image is None or mask is None:
                continue
            mask_main = cv2.resize(mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_masks.append(mask_main)

        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        aux_height = batch_masks.shape[1] // 16
        aux_width = batch_masks.shape[2] // 16
        batch_masks_aux = np.array([cv2.resize(m, (aux_width, aux_height), interpolation=cv2.INTER_NEAREST) for m in batch_masks])
        return batch_images, [batch_masks, batch_masks_aux]

def dice_loss(y_true, y_pred, smooth=1e-7):

    if isinstance(y_true, list):
        y_true = y_true[0]  
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES])

    mask = tf.not_equal(y_true_flat, 255)
    y_true_masked = tf.boolean_mask(y_true_flat, mask)
    y_pred_masked = tf.boolean_mask(y_pred_flat, mask)

    y_true_oh = tf.one_hot(tf.cast(y_true_masked, tf.int32), NUM_CLASSES)
    y_pred_masked = tf.cast(y_pred_masked, tf.float32)

    intersection = tf.reduce_sum(y_true_oh * y_pred_masked)
    union = tf.reduce_sum(y_true_oh) + tf.reduce_sum(y_pred_masked)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice

def auxiliary_loss(y_true, y_pred_aux, lambda_weight=0.3):

    y_true_resized = tf.image.resize(
        tf.expand_dims(tf.cast(y_true, tf.float32), -1),
        tf.shape(y_pred_aux)[1:3],
        method='nearest'
    )
    y_true_resized = tf.cast(tf.squeeze(y_true_resized, -1), tf.int32)
    
    aux_dice = dice_loss(y_true_resized, y_pred_aux)
    
    y_true_flat = tf.reshape(y_true_resized, [-1])
    y_pred_flat = tf.reshape(y_pred_aux, [-1, NUM_CLASSES])
    mask = tf.not_equal(y_true_flat, 255)
    y_true_masked = tf.boolean_mask(y_true_flat, mask)
    y_pred_masked = tf.boolean_mask(y_pred_flat, mask)
    aux_ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
    aux_ce = tf.reduce_mean(aux_ce)
    
    return lambda_weight * (aux_dice + aux_ce)

def combined_loss(y_true, y_pred):
    if isinstance(y_pred, list) and len(y_pred) == 2:
        y_pred_main, y_pred_aux = y_pred
        y_true_main, y_true_aux = y_true
        
        dice_main = dice_loss(y_true_main, y_pred_main)
        y_true_flat = tf.reshape(y_true_main, [-1])
        y_pred_flat = tf.reshape(y_pred_main, [-1, NUM_CLASSES])
        mask = tf.not_equal(y_true_flat, 255)
        y_true_masked = tf.boolean_mask(y_true_flat, mask)
        y_pred_masked = tf.boolean_mask(y_pred_flat, mask)
        ce_main = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
        loss_main =  ce_main + dice_main 

        aux_loss = auxiliary_loss(y_true_aux, y_pred_aux, lambda_weight=0.3)
        
        return loss_main + aux_loss
    else:
        if isinstance(y_true, list):
            y_true = y_true[0]  # Use main ground truth
            
        dice = dice_loss(y_true, y_pred)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES])
        mask = tf.not_equal(y_true_flat, 255)
        y_true_masked = tf.boolean_mask(y_true_flat, mask)
        y_pred_masked = tf.boolean_mask(y_pred_flat, mask)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
        loss_main =  ce + dice 
        return loss_main

def mean_iou(y_true, y_pred):

    if isinstance(y_true, list):
        y_true = y_true[0]  # Use main ground truth for metrics
        
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
        
        class_present = tf.reduce_sum(tf.cast(true_mask, tf.float32)) > 0
        iou = tf.where(tf.logical_and(union > 0, class_present), intersection / union, tf.constant(float('nan')))
        ious.append(iou)
    
    ious = tf.stack(ious)
    valid_ious = tf.boolean_mask(ious, ~tf.math.is_nan(ious))
    result = tf.reduce_mean(valid_ious)
    return result

def accuracy(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = y_true[0]  # Use main ground truth for metrics
        
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int64)
    mask = tf.not_equal(y_true, 255)
    matches = tf.equal(y_true, y_pred)
    matches = tf.logical_and(matches, mask)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(mask, tf.float32)
    acc = tf.reduce_sum(matches) / (tf.reduce_sum(mask) + 1e-6)
    return acc

class DecoderFreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_dec):
        super().__init__()
        self.freeze_dec = freeze_dec
        self.cycle_length = 10

    def on_epoch_begin(self, epoch, logs=None):
        if self.freeze_dec:
            cycle_position = epoch % self.cycle_length
            should_freeze = cycle_position < 7
            
            decoder_prefixes = [
                'dec', 'seghead_', 'output_softmax'
            ]
            for layer in self.model.layers:
                if layer.name.startswith('input_image') or layer.name.startswith('stem') or layer.name.startswith('enc'):
                    continue
                if any(layer.name.startswith(prefix) for prefix in decoder_prefixes):
                    layer.trainable = not should_freeze
            
class BatchNormFreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_bn):
        super().__init__()
        self.freeze_bn = freeze_bn

    def on_epoch_begin(self, epoch, logs=None):
        if self.freeze_bn:
            unfreeze = ((epoch) % 10 == 0)
            for layer in self.model.layers:
                if 'batchnorm' in layer.name.lower() or 'bn' in layer.name.lower():
                    layer.trainable = not self.freeze_bn or unfreeze
class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, save_dir, num_samples=4, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.save_frequency = save_frequency
        os.makedirs(save_dir, exist_ok=True)
        
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
        if (epoch + 1) <= 5 or ((epoch + 1) % self.save_frequency == 0):
            tf.config.experimental.disable_jit()
            saved = 0
            for batch_idx in range(min(8, len(self.val_generator))):
                if saved >= self.num_samples:
                    break
                batch = self.val_generator[batch_idx]
                batch_images, batch_masks = batch[0], batch[1][0] if isinstance(batch[1], list) else batch[1]
                batch_images, batch_masks = batch_images[:2], batch_masks[:2]
                
                with tf.device('/CPU:0'):
                    predictions = self.model(batch_images, training=False)
                    predictions = predictions[0] if isinstance(predictions, list) else predictions
                    predictions = predictions.numpy()
                
                for i in range(min(self.num_samples - saved, len(batch_images))):
                    os.makedirs(os.path.join(self.save_dir, f'sample_{saved + 1}'), exist_ok=True)
                    original_img = (batch_images[i] * 255).astype(np.uint8)
                    gt_mask = np.argmax(batch_masks[i], axis=-1) if batch_masks[i].ndim == 3 else batch_masks[i]
                    pred_mask = np.argmax(predictions[i], axis=-1) if predictions[i].ndim == 3 else predictions[i]
                    gt_colored = self.colorize_mask(gt_mask)
                    pred_colored = self.colorize_mask(pred_mask)
                    combined = np.hstack([original_img, gt_colored, pred_colored])
                    cv2.imwrite(os.path.join(self.save_dir, f'sample_{saved + 1}', f'epoch_{epoch+1:03d}.png'), 
                               cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    saved += 1
            tf.config.experimental.enable_jit()


class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, num_classes=NUM_CLASSES, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and hasattr(self.model.optimizer, 'learning_rate'):
            current_lr = float(self.model.optimizer.learning_rate)
            logs['learning_rate'] = current_lr

def create_callbacks(model_name, val_generator):

    callbacks = []
    
    viz_dir = os.path.join(MODELS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint_epoch_{{epoch:02d}}.weights.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_mean_iou',
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)


    def poly_decay(epoch):
        max_epochs = EPOCHS
        return LEARNING_RATE * (1 - epoch / max_epochs)
    
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
        num_samples=12,
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
    
    
    # Add BN freeze callback if requested
    if FREEZE_BN:
        callbacks.append(BatchNormFreezeCallback(FREEZE_BN))
    # Add decoder freeze callback if requested
    if FREEZE_DEC:
        callbacks.append(DecoderFreezeCallback(FREEZE_DEC))
        
    return callbacks

def load_compatible_weights(model, weights_path, skip_mismatched=True):
    model.load_weights(weights_path, by_name=True, skip_mismatch=skip_mismatched)
    return True

def convert_float32_to_qat(float32_model, config=MODEL_CONFIG_LMIA):

    qat_model = create_lmiinet(
        input_shape=float32_model.input_shape[1:],
        num_classes=config['num_classes'],
        config=config,
        use_quantization=True
    )

    f_layers = {l.name: l for l in float32_model.layers}
    q_layers = {l.name: l for l in qat_model.layers}
    
    # Store names of layers whose weights have been set
    processed_layers = set()

    for i, layer in enumerate(float32_model.layers):
        if layer.name in processed_layers:
            continue

        # Check for Conv/DepthwiseConv followed by BatchNormalization
        is_conv = isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
        
        if is_conv and i + 1 < len(float32_model.layers) and isinstance(float32_model.layers[i+1], tf.keras.layers.BatchNormalization):
            conv_layer = layer
            bn_layer = float32_model.layers[i+1]
            
            # Get corresponding Q-layers from the new QAT model
            q_conv_layer = q_layers.get(conv_layer.name)
            q_bn_layer = q_layers.get(bn_layer.name)

            if not q_conv_layer or not q_bn_layer:
                continue

            # Extract weights for folding
            conv_weights = conv_layer.get_weights()
            bn_weights = bn_layer.get_weights()

            kernel = conv_weights[0]
            bias = conv_weights[1] if len(conv_weights) > 1 else np.zeros(kernel.shape[-1])
            
            # BN weights: gamma, beta, moving_mean, moving_variance, (and renorm params if they exist)
            gamma, beta, mean, variance = bn_weights[0], bn_weights[1], bn_weights[2], bn_weights[3]
            epsilon = bn_layer.epsilon

            # The core folding calculation
            scale = gamma / np.sqrt(variance + epsilon)
            folded_kernel = kernel * scale
            folded_bias = (bias - mean) * scale + beta

            # Set the folded weights on the QConv layer
            q_conv_layer.set_weights([folded_kernel, folded_bias])

            # Reset the QBN layer to be an identity-like transform for QAT fine-tuning
            # This is critical because the normalization is now part of the convolution
            q_bn_gamma = np.ones_like(gamma)
            q_bn_beta = np.zeros_like(beta)
            q_bn_mean = np.zeros_like(mean)
            q_bn_variance = np.ones_like(variance)
            q_bn_layer.set_weights([q_bn_gamma, q_bn_beta, q_bn_mean, q_bn_variance])

            processed_layers.add(conv_layer.name)
            processed_layers.add(bn_layer.name)

        # Handle standalone layers (like Dense or Conv without a subsequent BN)
        elif layer.name in q_layers:
            q_layer = q_layers.get(layer.name)
            if q_layer and len(layer.get_weights()) > 0:
                q_layer.set_weights(layer.get_weights())
                processed_layers.add(layer.name)


    return qat_model

def main():

    initial_epoch = 0
    global USE_DROPOUT_DURING_TRAINING

    train_generator = CityscapesDataGenerator(
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=AUGMENTATION,
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


    if TRAIN_QAT:
        if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH or '')
            if 'qat' in os.path.basename(LOAD_MODEL_PATH).lower():

                if 'final' in os.path.basename(LOAD_MODEL_PATH).lower():
                    model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=True, config=MODEL_CONFIG_LMIA, aux_out=False)
                else:                    
                    model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=True, config=MODEL_CONFIG_LMIA, aux_out=True)
                model.load_weights(LOAD_MODEL_PATH)
                if m:
                    initial_epoch = int(m.group(1))
            else:
                float32_model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=False, config=MODEL_CONFIG_LMIA, aux_out=True)
                if load_compatible_weights(float32_model, LOAD_MODEL_PATH, skip_mismatched=True):
                    pass
                else:
                    pass
                model = convert_float32_to_qat(float32_model, config=MODEL_CONFIG_LMIA)
                if m:
                    initial_epoch = int(m.group(1))
        else:
            model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=True, config=MODEL_CONFIG_LMIA, aux_out=True)
    else:
        model = create_lmiinet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES, use_quantization=False, config=MODEL_CONFIG_LMIA, aux_out=True)
        if LOAD_MODEL_PATH:
            if load_compatible_weights(model, LOAD_MODEL_PATH, skip_mismatched=True):
                pass
            else:
                pass
            
            m = re.search(r'_epoch_(\d+)\.weights\.h5$', LOAD_MODEL_PATH)
            if m:
                initial_epoch = int(m.group(1))


    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    if isinstance(model.output, list):
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,  # Our custom loss handles both outputs
            metrics={
                'output_softmax': [accuracy, dice_loss, mean_iou],
                'aux_output_softmax': [accuracy, dice_loss, mean_iou]
            }
        )
    else:
        # Single output model (backward compatibility)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[accuracy, dice_loss, mean_iou]
        )

    # Print model summary
    model.summary()

    # Create callbacks
    model_name = "cgra4ml_lmiinet_qat" if TRAIN_QAT else "cgra4ml_lmiinet_baseline"
    callbacks = create_callbacks(model_name, val_generator)

    # Train model
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

    # Create final inference model (no dropout, no auxiliary output)
    
    # Temporarily disable dropout for inference model creation
    original_dropout_setting = USE_DROPOUT_DURING_TRAINING
    USE_DROPOUT_DURING_TRAINING = False
    
    # Create a new model with the same configuration but for inference
    final_model = create_lmiinet(
        input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), 
        num_classes=NUM_CLASSES, 
        use_quantization=TRAIN_QAT, 
        config=MODEL_CONFIG_LMIA,
        aux_out=False,  # Disable auxiliary output for inference
        inference_mode=True
    )
    
    USE_DROPOUT_DURING_TRAINING = original_dropout_setting
    
    for layer in final_model.layers:
        trained_layer = None
        for tl in model.layers:
            if tl.name == layer.name:
                trained_layer = tl
                break
        
        if trained_layer is not None and len(trained_layer.get_weights()) > 0:
            layer.set_weights(trained_layer.get_weights())
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
    final_model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[accuracy, dice_loss, mean_iou]
    )

    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.h5")
    final_model.save(final_model_path, save_format='h5')

    viz_dir_final = os.path.join(MODELS_DIR, 'visualizations_final')
    os.makedirs(viz_dir_final, exist_ok=True)
    final_viz_cb = PredictionVisualizationCallback(
        val_generator=val_generator,
        save_dir=viz_dir_final,
        num_samples=12,
        save_frequency=1
    )
    if hasattr(final_viz_cb, 'set_model'):
        final_viz_cb.set_model(final_model)
    else:
        final_viz_cb.model = final_model
    final_viz_cb.on_epoch_end(epoch=EPOCHS)

if __name__ == "__main__":
    main()
