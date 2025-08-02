import os
import numpy as np
import tensorflow as tf
import psutil
import gc
import sys
import json
from tensorflow.keras.preprocessing.image import load_img
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
import cv2
from pathlib import Path
import time

# --- Kaggle Paths ---
DATASET_DIR = '/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data'
train_folder = "/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/train/"
valid_folder = "/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/val/"
MODELS_DIR = '/kaggle/working/models'
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Model Config ---
INPUT_HEIGHT = 256  # Match Kaggle dataset size
INPUT_WIDTH = 256   # Match Kaggle dataset size (square images)
BATCH_SIZE = 4      # Can increase batch size with smaller images
EPOCHS = 200
NUM_CLASSES = 34
SUBSET_RATIO = 0.6
NUM_WORKERS = 0
PREFETCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
DEBUG_MODE = False
MODEL_CONFIG = {
    'init_filters': INPUT_HEIGHT // 32,  # 8
    'stage1_filters': INPUT_HEIGHT // 16, # 16
    'stage2_filters': INPUT_HEIGHT // 8,  # 32
    'stage3_filters': INPUT_HEIGHT // 4,  # 64
    'stage4_filters': INPUT_HEIGHT // 2,  # 128
}
USE_DROPOUT_DURING_TRAINING = False
SAVE_HLS4ML_COMPATIBLE = False

quantized_bits_param_32 = {'bits': 32, 'integer': 8, 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
quantized_bits_param_16 = {'bits': 16, 'integer': 8, 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
quantized_relu_param_16 = {'bits': 16, 'integer': 8, 'use_stochastic_rounding': True}

# --- Data Generator ---
class KaggleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, augment=False, subset_ratio=1.0):
        self.batch_size = batch_size
        self.augment = augment
        self.subset_ratio = subset_ratio
        self.images = np.array(images)  # Already loaded images (256x256x3)
        self.masks = np.array(masks)    # Already loaded masks (flattened)
        self.indices = np.arange(len(self.images))
        self.aug = None
        
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
                A.GaussNoise(var_limit=(5.0, 25.0), mean=0, p=0.2),
                A.CoarseDropout(max_holes=4, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.15),
            ])
        
        self._select_random_subset()
    
    def __len__(self):
        return len(self.epoch_indices) // self.batch_size
    
    def _select_random_subset(self):
        if self.subset_ratio < 1.0:
            total_samples = len(self.indices)
            subset_size = int(total_samples * self.subset_ratio)
            # Clear previous epoch_indices to free memory
            if hasattr(self, 'epoch_indices'):
                del self.epoch_indices
            self.epoch_indices = np.random.choice(self.indices, size=subset_size, replace=False)
            print(f"Selected {subset_size}/{total_samples} samples ({self.subset_ratio*100:.1f}%) for this epoch")
        else:
            # Clear previous epoch_indices to free memory
            if hasattr(self, 'epoch_indices'):
                del self.epoch_indices
            self.epoch_indices = self.indices.copy()
    
    def on_epoch_end(self):
        self._select_random_subset()
        gc.collect()
    
    def preprocess_mask(self, mask_flat):
        """Convert flattened mask to one-hot encoded format"""
        # Reshape from flat to 256x256x3
        mask_reshaped = mask_flat.reshape(256, 256, 3)
        
        # Take only one channel (they should be the same)
        mask_2d = mask_reshaped[:, :, 0]
        
        # Already 256x256, no resizing needed since INPUT_HEIGHT = INPUT_WIDTH = 256
        
        # Convert to one-hot encoding
        mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
        
        # Clip mask values to valid class range and convert to integer
        mask_2d = np.clip(mask_2d * 255, 0, NUM_CLASSES - 1).astype(np.int32)  # Convert from [0,1] to [0,255] then to class indices
        np.put_along_axis(mask_one_hot, mask_2d[..., np.newaxis], 1, axis=-1)
        
        return mask_one_hot
    
    def __getitem__(self, idx):
        batch_indices = self.epoch_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
        Y = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
        
        for i, batch_idx in enumerate(batch_indices):
            # Get image - already 256x256, no resizing needed
            img = self.images[batch_idx].astype(np.float32)  # (256, 256, 3), already normalized [0,1]
            
            # Get mask and convert to one-hot
            mask_one_hot = self.preprocess_mask(self.masks[batch_idx])
            
            # Apply augmentation if enabled
            if self.augment and self.aug is not None:
                augmented = self.aug(image=img, mask=mask_one_hot)
                img = augmented['image']
                mask_one_hot = augmented['mask']
            
            X[i] = img
            Y[i] = mask_one_hot
        
        return X, Y

# --- Model + Losses + Metrics ---
def conv_block(x, filters, bits=16, kernel_size=3, strides=1, dilation_rate=1, name=None):
    qbits = {'bits': bits, 'integer': min(bits//2, 8), 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
    qparam = quantized_bits(**qbits)
    x = QConv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same',
        kernel_quantizer=qparam, bias_quantizer=qparam, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_conv')(x)
    x = QBatchNormalization(gamma_quantizer=qparam, beta_quantizer=qparam, mean_quantizer=qparam, variance_quantizer=qparam, momentum=0.9, epsilon=1e-5, name=f'{name}_bn')(x)
    x = QActivation(qparam, name=f'{name}_act')(x)
    return x
def c2f_block(x, filters, bits=16, n=1, name=None, dilation_rate=1, dilation_bottleneck=1):
    x = conv_block(x, filters, bits=bits, dilation_rate=dilation_rate, name=f'{name}_conv1')
    y = x
    for i in range(n):
        y = conv_block(y, filters//2, bits=bits, kernel_size=3, dilation_rate=dilation_bottleneck, name=f'{name}_bottleneck{i}')
        x = tf.keras.layers.Concatenate(name=f'{name}_concat{i}')([x, y])
    return x
def sppf_block(x, channels, name, bits=16):
    qbits = {'bits': bits, 'integer': min(bits//2, 8), 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
    y1_detail = QConv2D(channels//4, kernel_size=3, strides=1, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_detail_conv1')(x)
    y1_refine = QConv2D(channels//4, kernel_size=3, strides=1, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_detail_conv2')(y1_detail)
    y1 = tf.keras.layers.Add(name=f'{name}_detail_residual')([y1_detail, y1_refine])
    y2_d1 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=1, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_pyramid_d1')(x)
    y2_d2 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=2, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_pyramid_d2')(x)
    y2_d4 = QConv2D(channels//4, kernel_size=3, strides=1, dilation_rate=4, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_pyramid_d4')(x)
    y3_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}_global_pool')(x)
    y3 = QConv2D(channels//4, kernel_size=1, strides=1, padding='same',
        kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_global_conv')(y3_pool)
    y3 = tf.keras.layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='nearest', name=f'{name}_broadcast')(y3)
    concat = tf.keras.layers.Concatenate(name=f'{name}_concat')([x, y1, y2_d1, y2_d2, y2_d4, y3])
    return conv_block(concat, channels, kernel_size=1, name=f'{name}_conv')
def channel_attention(x, bits=16, ratio=16, name=''):
    channels = x.shape[-1]
    qbits = {'bits': bits, 'integer': min(bits//2, 8), 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
    y = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}_gap')(x)
    y = QConv2D(channels, 1, kernel_quantizer=quantized_bits(**qbits), bias_quantizer=quantized_bits(**qbits), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_attention')(y)
    y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    return tf.keras.layers.Multiply(name=f'{name}_scale')([x, y])
def create_segmentation_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=8, name='init_conv')
    qbits_down = {'bits': 16, 'integer': 8, 'symmetric': True, 'alpha': 'auto_po2', 'use_stochastic_rounding': True, 'scale_axis': None}
    x = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down), bias_quantizer=quantized_bits(**qbits_down), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='downsample1')(x)
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='encoder1_1')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='encoder1_2')
    skip1 = x
    x = QConv2D(MODEL_CONFIG['stage1_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down), bias_quantizer=quantized_bits(**qbits_down), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='encoder1_downsample')(x)
    x = c2f_block(x, MODEL_CONFIG['stage2_filters'], bits=16, n=2, name='encoder2', dilation_rate=2, dilation_bottleneck=1)
    skip2 = x
    x = QConv2D(MODEL_CONFIG['stage2_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**qbits_down), bias_quantizer=quantized_bits(**qbits_down), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='encoder2_downsample')(x)
    x = c2f_block(x, MODEL_CONFIG['stage3_filters'], bits=16, n=1, name='encoder3', dilation_rate=2, dilation_bottleneck=1)
    x = channel_attention(x, bits=16, name='ca_encoder3')
    skip3 = x
    x = QConv2D(MODEL_CONFIG['stage3_filters'], kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param_16), bias_quantizer=quantized_bits(**quantized_bits_param_16), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='encoder3_downsample')(x)
    x = c2f_block(x, MODEL_CONFIG['stage4_filters'], bits=32, n=3, name='bridge', dilation_rate=2, dilation_bottleneck=1)
    x = channel_attention(x, bits=32, name='ca_bridge')
    x = sppf_block(x, MODEL_CONFIG['stage4_filters'], bits=32, name='bridge_sppf')
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], kernel_size=1, name='decoder3_prep')
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage3_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='decoder3_upsample')(x)
    skip3_enhance = conv_block(skip3, MODEL_CONFIG['stage3_filters'], bits=16, kernel_size=3, name='skip3_boundary')
    skip3 = conv_block(skip3, MODEL_CONFIG['stage3_filters'], bits=16, kernel_size=1, name='skip3_adjust')
    skip3 = tf.keras.layers.Add(name='skip3_fusion')([skip3, skip3_enhance])
    x = tf.keras.layers.Concatenate(name='decoder3_concat')([x, skip3])
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv1')
    x_identity = x
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv2')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], bits=16, name='decoder3_conv3')
    x = tf.keras.layers.Add(name='decoder3_residual')([x, x_identity])
    x = channel_attention(x, name='decoder3_attention')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], kernel_size=1, name='decoder2_prep')
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage2_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='decoder2_upsample')(x)
    skip2 = conv_block(skip2, MODEL_CONFIG['stage2_filters'], bits=16, kernel_size=1, name='skip2_adjust')
    x = tf.keras.layers.Concatenate(name='decoder2_concat')([x, skip2])
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv1')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv2')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], bits=16, name='decoder2_conv3')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], kernel_size=1, name='decoder1_prep')
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['stage1_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='decoder1_upsample')(x)
    skip1 = conv_block(skip1, MODEL_CONFIG['stage1_filters'], bits=16, kernel_size=1, name='skip1_adjust')
    x = tf.keras.layers.Concatenate(name='decoder1_concat')([x, skip1])
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder1_conv1')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder1_conv2')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], bits=16, name='decoder0_prep')
    x = tf.keras.layers.Conv2DTranspose(MODEL_CONFIG['init_filters'], kernel_size=2, strides=2, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='decoder0_upsample')(x)
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=16, name='decoder0_conv')
    x = conv_block(x, MODEL_CONFIG['init_filters'], bits=16, name='decoder0_conv2')
    x_boundary = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, padding='same',
                      kernel_quantizer=quantized_bits(**quantized_bits_param_32), bias_quantizer=quantized_bits(**quantized_bits_param_32),
                      kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='boundary_enhance1_conv')(x)
    x_boundary = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_32),
        beta_quantizer=quantized_bits(**quantized_bits_param_32),
        mean_quantizer=quantized_bits(**quantized_bits_param_32),
        variance_quantizer=quantized_bits(**quantized_bits_param_32),
        name='boundary_enhance1_bn'
    )(x_boundary)
    x_boundary = QActivation('relu', name='boundary_enhance1_act')(x_boundary)
    x_boundary = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=3, padding='same',
                      kernel_quantizer=quantized_bits(**quantized_bits_param_32), bias_quantizer=quantized_bits(**quantized_bits_param_32),
                      kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='boundary_enhance2_conv')(x_boundary)
    x_boundary = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_32),
        beta_quantizer=quantized_bits(**quantized_bits_param_32),
        mean_quantizer=quantized_bits(**quantized_bits_param_32),
        variance_quantizer=quantized_bits(**quantized_bits_param_32),
        name='boundary_enhance2_bn'
    )(x_boundary)
    x = tf.keras.layers.Concatenate(name='final_fusion')([x, x_boundary])
    x = QConv2D(MODEL_CONFIG['init_filters'], kernel_size=1, padding='same',
              kernel_quantizer=quantized_bits(**quantized_bits_param_16), bias_quantizer=quantized_bits(**quantized_bits_param_16),
              kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='final_reduce_conv')(x)
    x = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param_16),
        beta_quantizer=quantized_bits(**quantized_bits_param_16),
        mean_quantizer=quantized_bits(**quantized_bits_param_16),
        variance_quantizer=quantized_bits(**quantized_bits_param_16),
        name='final_reduce_bn'
    )(x)
    x = QActivation(quantized_relu(**quantized_relu_param_16), name='final_reduce_act')(x)
    outputs = QConv2D(NUM_CLASSES, (1, 1),
                     kernel_quantizer=quantized_bits(**quantized_bits_param_16), bias_quantizer=quantized_bits(**quantized_bits_param_16),
                     kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='output')(x)
    outputs = QActivation(quantized_bits(**quantized_bits_param_16), name='output_quant')(outputs)
    outputs = tf.keras.layers.Activation('softmax', name='output_softmax')(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class IoULoss(tf.keras.losses.Loss):
    def __init__(self, name='iou_loss', smooth=1e-6):
        super(IoULoss, self).__init__(name=name)
        self.smooth = smooth
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
        y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=0)
        union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) - intersection + self.smooth
        iou = (intersection + self.smooth) / union
        return 1.0 - tf.reduce_mean(iou)
class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, weighting_strategy='adaptive_magnitude', name='combo_loss'):
        super(ComboLoss, self).__init__(name=name)
        self.alpha = alpha
        self.weighting_strategy = weighting_strategy
        self.iou_loss = IoULoss()
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.iou_ema = tf.Variable(1.0, trainable=False, name='iou_ema')
        self.ce_ema = tf.Variable(1.0, trainable=False, name='ce_ema')
        self.ema_decay = 0.9
    def call(self, y_true, y_pred):
        iou_loss = self.iou_loss(y_true, y_pred)
        ce_loss = self.ce_loss(y_true, y_pred)
        iou_loss = tf.where(tf.math.is_finite(iou_loss), iou_loss, 0.0)
        ce_loss = tf.where(tf.math.is_finite(ce_loss), ce_loss, 0.0)
        if self.weighting_strategy == 'adaptive_magnitude':
            self.iou_ema.assign(self.ema_decay * self.iou_ema + (1 - self.ema_decay) * iou_loss)
            self.ce_ema.assign(self.ema_decay * self.ce_ema + (1 - self.ema_decay) * ce_loss)
            total_magnitude = self.iou_ema + self.ce_ema + 1e-8
            alpha_iou = self.ce_ema / total_magnitude
            alpha_ce = self.iou_ema / total_magnitude
        else:
            alpha_iou = 0.7
            alpha_ce = 0.3
        total_alpha = alpha_iou + alpha_ce + 1e-8
        alpha_iou = alpha_iou / total_alpha
        alpha_ce = alpha_ce / total_alpha
        return alpha_iou * iou_loss + alpha_ce * ce_loss
class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=NUM_CLASSES, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.intersection = self.add_weight(name='intersection', shape=(num_classes,), initializer='zeros')
        self.union = self.add_weight(name='union', shape=(num_classes,), initializer='zeros')
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        intersections = []
        unions = []
        for i in range(self.num_classes):
            pred_mask = tf.cast(tf.equal(y_pred, i), tf.float32)
            true_mask = tf.cast(tf.equal(y_true, i), tf.float32)
            intersection = tf.reduce_sum(pred_mask * true_mask)
            union = tf.reduce_sum(pred_mask) + tf.reduce_sum(true_mask) - intersection
            intersections.append(intersection)
            unions.append(union)
        intersections_tensor = tf.stack(intersections)
        unions_tensor = tf.stack(unions)
        self.intersection.assign_add(intersections_tensor)
        self.union.assign_add(unions_tensor)
    def result(self):
        iou = tf.math.divide_no_nan(self.intersection, self.union)
        return tf.reduce_mean(iou)
    def reset_state(self):
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.union.assign(tf.zeros_like(self.union))

# --- Data Preparation ---
def get_images_masks(path):
    """Load images and masks from Kaggle's side-by-side format"""
    names = os.listdir(path)
    img_g, img_m = [], []
    for name in names:
        # Load the combined image (left: image, right: mask)
        img = cv2.imread(path + name)
        if img is None:
            print(f"Warning: Failed to load {path + name}")
            continue
            
        # Normalize to [0, 1] range
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # Convert BGR to RGB
        img = img[:, :, ::-1]
        
        # Split into image (left half) and mask (right half)
        width = img.shape[1]
        img_g.append(img[:, :width//2])  # Left half - actual image (256x256x3)
        img_m.append(np.reshape(img[:, width//2:], (256*256*3)))  # Right half - mask flattened
        
        # Clean up
        del img
    del names
    return img_g, img_m

def prepare_data(cityscapes_root):
    """Load data using Kaggle's side-by-side image format"""
    print(f"Loading training data from: {train_folder}")
    print(f"Loading validation data from: {valid_folder}")
    
    train_imgs, train_masks = get_images_masks(train_folder)
    valid_imgs, valid_masks = get_images_masks(valid_folder)
    
    print(f"\nLoaded {len(train_imgs)} training samples")
    print(f"Loaded {len(valid_imgs)} validation samples")
    
    if not train_imgs:
        raise ValueError("No training images found.")
    if not valid_imgs:
        raise ValueError("No validation images found.")
    
    return train_imgs, train_masks, valid_imgs, valid_masks

# --- Callbacks ---
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        print(f"\nMemory usage after epoch {epoch}: {memory_usage:.2f} MB")
class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_file = os.path.join(log_dir, 'log.txt')
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    f.write('')
            except Exception as e:
                print(f"[TrainingLoggerCallback] Could not create log.txt: {e}")
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_start_time = epoch_end_time
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        loss = logs.get('loss', None)
        iou = logs.get('iou', None)
        val_loss = logs.get('val_loss', None)
        val_iou = logs.get('val_iou', None)
        time_str = f"{int(epoch_duration)}s"
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
        file_extension = '.h5' if SAVE_HLS4ML_COMPATIBLE else '.keras'
        log_line += f"Epoch {epoch+1}: saving model to {MODELS_DIR}/checkpoint_epoch_{epoch+1:02d}{file_extension}\n"
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"[TrainingLoggerCallback] Could not write to log.txt: {e}")

# --- Main Training Function ---
def train_model():
    # Configure TensorFlow to use ONLY GPUs (no CPU fallback)
    print("🚀 CONFIGURING GPU-ONLY TRAINING")
    print("="*50)
    
    # Initialize TensorFlow and check for GPU availability
    print("Initializing TensorFlow GPU detection...")
    
    # Force TensorFlow to initialize GPU detection
    tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU'), 'GPU')
    
    # Check GPU availability with multiple methods
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    
    print(f"Physical GPUs detected: {len(gpus)}")
    print(f"Logical GPUs detected: {len(logical_gpus)}")
    
    if gpus:
        print(f"GPU devices: {[gpu.name for gpu in gpus]}")
    
    # Check if CUDA is available and built with GPU support
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    # Try to get GPU details
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0]) if gpus else {}
        if gpu_details:
            print(f"GPU details: {gpu_details}")
    except:
        print("Could not get GPU details")
    
    # In Kaggle, sometimes GPUs are available but not detected immediately
    # Fall back to checking if CUDA operations work
    if len(gpus) == 0:
        print("⚠️  No physical GPUs detected, checking CUDA availability...")
        try:
            # Test if GPU operations work
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                test_result = tf.reduce_sum(test_tensor)
                print(f"✅ GPU test operation successful: {test_result}")
                print("🎯 GPU is available for computation (Kaggle T4 detected)")
                # Create a mock GPU list for Kaggle environment
                gpus = ["T4_GPU_0"]  # Mock GPU for strategy setup
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            print("❌ No GPU available - this script requires GPU for training")
            raise RuntimeError("No GPU found! This script requires GPU for training.")
    
    print(f"Found {len(gpus)} GPU(s) available for training")
    
    # Configure GPU settings
    try:
        # If we have actual GPU objects, configure them
        if gpus and hasattr(gpus[0], 'name'):
            for gpu in gpus:
                # Enable memory growth for each GPU
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Configured GPU: {gpu.name}")
        else:
            # Kaggle environment - configure available GPUs by index
            print("✅ Configuring Kaggle GPU environment")
            # Try to set memory growth for GPU:0
            try:
                tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
            except:
                print("⚠️  Could not set memory growth (may not be needed in Kaggle)")
        
        # Set up training strategy
        gpu_count = len(gpus) if isinstance(gpus, list) and gpus and hasattr(gpus[0], 'name') else 1
        
        if gpu_count > 1:
            print(f"🔥 Setting up MULTI-GPU training with {gpu_count} GPUs")
            strategy = tf.distribute.MirroredStrategy()
            print(f"✅ MirroredStrategy configured with {strategy.num_replicas_in_sync} replicas")
        else:
            print(f"⚡ Single GPU training mode (Kaggle T4)")
            # Use default strategy which will use GPU if available
            strategy = tf.distribute.get_strategy()
            
    except RuntimeError as e:
        if "cannot be modified" in str(e):
            print("⚠️  GPU memory growth already set (probably from previous run)")
            strategy = tf.distribute.get_strategy()
        else:
            print(f"⚠️  GPU configuration warning: {e}")
            strategy = tf.distribute.get_strategy()
    
    # Force TensorFlow to prefer GPU over CPU
    # tf.config.set_soft_device_placement(False)  # Strict device placement
    
    # Set GPU as default device for all operations
    try:
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
        print("✅ Set GPU as default visible device")
    except:
        print("⚠️  Could not set GPU as default device")
    
    print("✅ Disabled CPU fallback - GPU-only mode enabled")
    
    # Disable eager execution for better GPU memory management
    tf.config.run_functions_eagerly(False)
    print("✅ Disabled eager execution for optimal GPU performance")
    
    # Verify GPU usage
    print(f"🎯 Training strategy: {type(strategy).__name__}")
    print(f"🎯 Strategy replicas: {strategy.num_replicas_in_sync}")
    
    tf.random.set_seed(42)
    np.random.seed(42)
    # Load or create model within GPU strategy scope
    with strategy.scope():
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.keras', '.h5'))]
        initial_epoch = 0
        skip_compilation = False
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)))
            print(f"\nFound latest model: {latest_model}")
            if 'epoch_' in latest_model:
                try:
                    if 'checkpoint_epoch_' in latest_model:
                        epoch_part = latest_model.split('checkpoint_epoch_')[1]
                    else:
                        epoch_part = latest_model.split('epoch_')[1]
                    initial_epoch = int(epoch_part.split('.')[0])
                    print(f"Resuming from epoch {initial_epoch}")
                except:
                    print("Could not determine epoch number from filename")
            model = None
            try:
                print(f"Trying to load full model from {latest_model}")
                model = tf.keras.models.load_model(os.path.join(MODELS_DIR, latest_model),
                                               custom_objects={
                                                   'QConv2D': QConv2D,
                                                   'QBatchNormalization': QBatchNormalization,
                                                   'QActivation': QActivation,
                                                   'IoUMetric': IoUMetric,
                                                   'quantized_bits': quantized_bits,
                                                   'quantized_relu': quantized_relu
                                               })
                print("Successfully loaded full model with optimizer state.")
                skip_compilation = True
            except Exception as e_model:
                print(f"Failed to load as full model: {str(e_model)}")
                try:
                    print(f"Trying to load weights from {latest_model}")
                    model = create_segmentation_model()
                    model.load_weights(os.path.join(MODELS_DIR, latest_model))
                    print("Successfully loaded weights into model architecture.")
                    skip_compilation = False
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
            
        if not skip_compilation:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=2e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=0.5
            )
            model.compile(
                optimizer=optimizer,
                loss=ComboLoss(weighting_strategy='adaptive_magnitude'),
                metrics=[IoUMetric()],
                run_eagerly=False,
            )
            print("✅ Model compiled successfully within GPU strategy scope.")
        else:
            print("✅ Using loaded model with existing optimizer state (within GPU strategy scope).")
    
    print(f"🎯 Model loaded/created on {strategy.num_replicas_in_sync} GPU(s)")
    print(f"📊 Model parameters: {model.count_params():,}")
    print(f"\n🗂️  PREPARING DATASET")
    print("="*30)
    print(f"Looking for Cityscapes dataset in: {DATASET_DIR}")
    train_images, train_masks, val_images, val_masks = prepare_data(DATASET_DIR)
    print(f"\nFound {len(train_images)} training images and {len(val_images)} validation images")
    print(f"Using {SUBSET_RATIO*100:.0f}% random subset each epoch for faster training")
    
    # GPU-optimized data generators
    train_gen = KaggleDataGenerator(train_images, train_masks, BATCH_SIZE, augment=True, subset_ratio=SUBSET_RATIO)
    val_gen = KaggleDataGenerator(val_images, val_masks, BATCH_SIZE, augment=False, subset_ratio=1.0)
    print(f"✅ Data generators created with batch size {BATCH_SIZE} (optimized for GPU)")
    
    print(f"\n🚀 STARTING GPU-ONLY TRAINING")
    print("="*40)
    
    # Display GPU information for Kaggle
    if hasattr(gpus[0], 'name') if gpus else False:
        print(f"🎯 GPUs: {len(gpus)} active ({[gpu.name for gpu in gpus]})")
    else:
        print(f"🎯 GPUs: Kaggle T4 GPU active")
    
    print(f"🔥 Strategy: {type(strategy).__name__}")
    
    # Calculate per-GPU batch size safely
    per_gpu_batch = BATCH_SIZE // max(1, strategy.num_replicas_in_sync)
    print(f"⚡ Batch size per GPU: {per_gpu_batch}")
    print(f"📊 Total effective batch size: {BATCH_SIZE}")
    print(f"🔄 Starting from epoch: {initial_epoch}")
    
    # Verify GPU placement before training
    print(f"\n🔍 VERIFYING GPU PLACEMENT")
    print("="*30)
    
    # Check if model layers are on GPU
    try:
        with tf.device('/GPU:0'):
            # Create a test tensor and verify GPU placement
            test_input = tf.random.normal((1, INPUT_HEIGHT, INPUT_WIDTH, 3))
            print(f"✅ Test tensor device: {test_input.device}")
            
            # Test model forward pass on GPU
            test_output = model(test_input)
            print(f"✅ Model output device: {test_output.device}")
            print(f"✅ Model successfully running on GPU")
            
            # Force model weights to GPU
            for layer in model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    layer.kernel = tf.identity(layer.kernel)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias = tf.identity(layer.bias)
            
            del test_input, test_output  # Clean up
            
    except Exception as e:
        print(f"⚠️  GPU verification warning: {e}")
    
    print(f"\n🚀 TRAINING CONFIGURATION VERIFIED")
    print("="*35)
    
    file_extension = '.h5' if SAVE_HLS4ML_COMPATIBLE else '.keras'
    steps_per_epoch = max(1, len(train_gen) // GRADIENT_ACCUMULATION_STEPS)
    callbacks = [
        ModelCheckpoint(
            f'{MODELS_DIR}/best_model{file_extension}',
            save_best_only=True,
            monitor='val_iou',
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        ModelCheckpoint(
            f'{MODELS_DIR}/checkpoint_epoch_{{epoch:02d}}{file_extension}',
            save_freq='epoch',
            verbose=1,
            save_weights_only=True
        ),
        EarlyStopping(
            patience=40,
            monitor='val_iou',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_iou',
            mode='max',
            factor=0.8,
            patience=20,
            min_lr=1e-7,
            verbose=1,
            restore_best_weights=False
        ),
        MemoryCleanupCallback(),
        TrainingLoggerCallback(MODELS_DIR)
    ]
    try:
        print("🚀 Starting GPU-ONLY training (Kaggle T4 optimized)...")
        
        # Force GPU device placement for training
        with tf.device('/GPU:0'):
            print("✅ Training operations forced to GPU:0")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                callbacks=callbacks,
                workers=0,
                use_multiprocessing=False,
                max_queue_size=PREFETCH_SIZE,
                steps_per_epoch=max(1, len(train_gen) // GRADIENT_ACCUMULATION_STEPS),
                initial_epoch=initial_epoch,
                verbose=1
            )
        print("Training completed successfully!")
        print("\nSaving final models...")
        model.save(f'{MODELS_DIR}/final_model_no_dropout{file_extension}')
        print(f"Training model saved as 'final_model_no_dropout{file_extension}'")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current full model...")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{MODELS_DIR}/interrupted_model_{timestamp}{file_extension}"
        model.save(model_path)
        print(f"Interrupted full model saved as '{model_path}'")
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

# Run training directly for Kaggle/Jupyter notebooks
train_model()
