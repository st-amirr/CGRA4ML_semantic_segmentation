#!/usr/bin/env python3
"""
HLS4ML Semantic Segmentation Training with QAT Toggle
=====================================================

This script implements the recommended hls4ml approach:
1. Train a baseline model in float32 (trainQAT=False)
2. Load the trained float32 model and apply QAT (trainQAT=True)

Usage:
    python train_qat_toggle.py --trainQAT=False  # Train baseline float32 model
    python train_qat_toggle.py --trainQAT=True   # Apply QAT to trained model

Key Features:
- Baseline float32 training for optimal convergence
- QAT application after float32 training completes
- Same architecture with quantized/non-quantized layer swapping
- Memory-efficient data pipeline
- FPGA-compatible architecture (no MaxPooling, etc.)

Recommended training command for improved results (reduces scattered pixels):
Key improvements:
- Combined loss function (categorical crossentropy + dice loss)
- Enhanced data augmentation with geometric and color transformations
- Post-processing callback to clean scattered pixels
- More conservative learning rate scheduling
- Lower initial learning rate for stability

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training tensorboard --logdir="/app/models/tb_logs" --host=0.0.0.0 --port=6006

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=100 --batch_size=4 --use_dropout=True --subset_ratio=0.9 
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=100 --batch_size=3 --use_dropout=True --subset_ratio=0.9 --LR=1e-5
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=False --epochs=100 --batch_size=3 --use_dropout=False --subset_ratio=0.9 --LR=1e-5

Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 /workspace/trainQAT.py --trainQAT=True --load_model=./models/float32_model_checkpoint_epoch_70.keras --epochs=30 --batch_size=2 --use_dropout=False --subset_ratio=0.9 --LR=1e-7

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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Multiply, Dropout, Dense
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu, QDense
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow.keras.backend as K


# Parse command line arguments
parser = argparse.ArgumentParser(description='HLS4ML Semantic Segmentation with QAT Toggle')
parser.add_argument('--trainQAT', type=str, default='False', 
                   help='Enable QAT training (True/False). Default: False for float32 baseline')
parser.add_argument('--load_model', type=str, default='', 
                   help='Path to float32 model to load for QAT training')
parser.add_argument('--epochs', type=int, default=50, 
                   help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2, 
                   help='Batch size for training')
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

# Define constants
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 34
DATASET_DIR = '/data'
NUM_WORKERS = 4
PREFETCH_MAX = 8

DEBUG_MODE = False

# Configure TensorFlow
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)
tf.config.set_soft_device_placement(True)

# Configure GPU to use more VRAM (from train.py)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Allow TensorFlow to allocate more GPU memory
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set memory limit to use more VRAM (10GB limit for better utilization)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*10)]  # 10GB limit
            )
    except RuntimeError as e:
        print(f"GPU memory growth setup failed: {e}")

# Enable additional TensorFlow optimizations for GPU utilization
tf.config.optimizer.set_jit(True)  # Enable XLA compilation for better GPU utilization
# tf.config.experimental.enable_op_determinism(False)  # Allow non-deterministic ops for performance

print(f"Eager execution enabled: {tf.executing_eagerly()}")
tf.debugging.set_log_device_placement(DEBUG_MODE)

# Model architecture configuration
MODEL_CONFIG = {
    'init_filters': INPUT_HEIGHT // 32,  # 16
    'stage1_filters': INPUT_HEIGHT // 16, # 32
    'stage2_filters': INPUT_HEIGHT // 8,  # 64
    'stage3_filters': INPUT_HEIGHT // 4,  # 128
    'stage4_filters': INPUT_HEIGHT // 2,  # 256
}

# Quantization configurations
quantized_bits_8_config = {
    'bits': 8,
    'integer': 1,  # for values within ±2 range
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# For Kernels and Biases with medium range (~1 to ~4): final_reduce_bn (1.903646), boundary_enhance2_bn (1.524222)
quantized_bits_16_config = {
    'bits': 8,
    'integer': 2,  # for values within ±4 range
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# For Output Layer and large range layers: already set correctly
quantized_bits_32_config = {
    'bits': 16,
    'integer': 8,  # already set correctly
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}

# General activations: safely increase integer bits to avoid saturation or overflow
quantized_relu_config = {
    'bits': 8,
    'integer': 6,  # increase from 4 to 6 (max activation ~±64)
    'use_stochastic_rounding': True
}

# Disable layout optimizer to prevent conflicts with QKeras
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
})

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
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_coeff

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
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Clip predictions to prevent log(0)
    y_pred_f = K.clip(y_pred_f, K.epsilon(), 1.0 - K.epsilon())
    
    # Calculate focal loss
    pt = y_true_f * y_pred_f + (1 - y_true_f) * (1 - y_pred_f)
    focal_loss = -alpha * K.pow(1 - pt, gamma) * K.log(pt)
    return K.mean(focal_loss)


def weighted_combined_loss(y_true, y_pred):
    """
    Combined categorical crossentropy, dice loss, and focal loss
    Best of all three worlds for class imbalance and segmentation
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return 0.4 * ce + 0.5 * dice + 0.1 * focal

# Add custom metrics for better monitoring
def mean_iou(y_true, y_pred):
    """
    Mean IoU metric for semantic segmentation - corrected to match calculate_iou logic
    """
    # Convert to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    # Calculate IoU for each class
    ious = []
    for class_id in range(NUM_CLASSES):
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

# Class-specific IoU metrics for Cityscapes classes
def cars(y_true, y_pred):
    """IoU metric for car class (class index 13)"""
    return class_iou(y_true, y_pred, class_id=13)

def road(y_true, y_pred):
    """IoU metric for road class (class index 0)"""
    return class_iou(y_true, y_pred, class_id=0)

def sidewalk(y_true, y_pred):
    """IoU metric for sidewalk class (class index 1)"""
    return class_iou(y_true, y_pred, class_id=1)

def traffic_light(y_true, y_pred):
    """IoU metric for traffic_light class (class index 6)"""
    return class_iou(y_true, y_pred, class_id=6)

def traffic_sign(y_true, y_pred):
    """IoU metric for traffic_sign class (class index 7)"""
    return class_iou(y_true, y_pred, class_id=7)

def class_iou(y_true, y_pred, class_id):
    """Generic IoU metric for a specific class"""
    # Convert to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Calculate IoU for the specific class
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), tf.float32))
    
    # Handle edge cases
    iou = tf.where(union > 0, intersection / union,
                   tf.where(intersection == 0, 1.0, 0.0))  # 1.0 if both intersection and union are 0
    
    return iou

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
        
        # Cityscapes color palette (simplified version)
        self.colors = [
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk  
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ]
        # Extend colors for all 34 classes
        while len(self.colors) < NUM_CLASSES:
            self.colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    def colorize_mask(self, mask):
        """Convert class indices to RGB colors"""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(min(len(self.colors), NUM_CLASSES)):
            colored_mask[mask == class_id] = self.colors[class_id]
        return colored_mask
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            try:
                # Get a batch of validation data
                batch_images, batch_masks = self.val_generator[0]
                
                # Make predictions
                predictions = self.model.predict(batch_images, verbose=0)
                
                # Process samples
                for i in range(min(self.num_samples, len(batch_images))):
                    # Original image
                    original_img = (batch_images[i] * 255).astype(np.uint8)
                    
                    # Ground truth mask
                    gt_mask = np.argmax(batch_masks[i], axis=-1)
                    gt_colored = self.colorize_mask(gt_mask)
                    
                    # Predicted mask
                    pred_mask = np.argmax(predictions[i], axis=-1)
                    pred_colored = self.colorize_mask(pred_mask)
                    
                    # Create combined visualization
                    combined = np.hstack([original_img, gt_colored, pred_colored])
                    
                    # Save as image
                    save_path = os.path.join(self.save_dir, f'epoch_{epoch+1:03d}_sample_{i+1}.png')
                    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                
                print(f"\n📸 Saved prediction visualizations for epoch {epoch+1} in {self.save_dir}")
                
            except Exception as e:
                print(f"⚠️  Could not save prediction visualizations: {e}")

class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, save_dir, num_classes=NUM_CLASSES, save_frequency=5):
        super().__init__()
        self.val_generator = val_generator
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.save_frequency = save_frequency
        os.makedirs(save_dir, exist_ok=True)
        

    
    def on_epoch_end(self, epoch, logs=None):
        try:
            # Learning rate tracking (this is still useful since it's not automatically logged)
            if logs is not None and hasattr(self.model.optimizer, 'learning_rate'):
                current_lr = float(self.model.optimizer.learning_rate)
                logs['learning_rate'] = current_lr
                
        except Exception as e:
            print(f"⚠️  Error in DetailedMetricsCallback: {e}")


class CityscapesDataGenerator(tf.keras.utils.Sequence):
    """Data generator for Cityscapes dataset"""
    
    def __init__(self, image_dir, mask_dir, batch_size=1, subset_ratio=1.0, 
                 target_height=512, target_width=1024, shuffle=True, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.shuffle = shuffle
        self.augment = augment
        
        # Get all image paths
        self.image_paths = list(self.image_dir.glob('**/*.png'))
        if subset_ratio < 1.0:
            subset_size = int(len(self.image_paths) * subset_ratio)
            self.image_paths = self.image_paths[:subset_size]
        
        print(f"Found {len(self.image_paths)} images")
        
        # Create balanced augmentation pipeline with reduced probabilities
        if self.augment:
            self.transform = A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=30, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, p=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
                
                # Color and contrast augmentations
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
                
                # Noise and blur augmentations
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.15),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.MotionBlur(blur_limit=7, p=0.1),
                
                # Weather and lighting effects
                A.RandomRain(drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.7, rain_type='drizzle', p=0.05),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.05),
                
                # Occlusion and masking
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.1),
            ])
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
            
            # Load corresponding mask
            mask_name = image_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            mask_path = self.mask_dir / image_path.parent.name / mask_name
            
            if not mask_path.exists():
                # Try alternative mask naming
                mask_name = image_path.name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                mask_path = self.mask_dir / image_path.parent.name / mask_name
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            # Resize image and mask
            image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
            
            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Convert mask to one-hot encoding for categorical crossentropy
            mask = tf.keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            dummy_image = np.zeros((self.target_height, self.target_width, 3), dtype=np.float32)
            dummy_mask = np.zeros((self.target_height, self.target_width, NUM_CLASSES), dtype=np.float32)
            dummy_mask[:, :, 0] = 1  # Set background class
            return dummy_image, dummy_mask
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for path in batch_paths:
            image, mask = self.load_and_preprocess_image(path)
            batch_images.append(image)
            batch_masks.append(mask)
        
        return np.array(batch_images), np.array(batch_masks)

def conv_block(x, filters, kernel_size=3, strides=1, use_quantization=False, precision='8bit', name='conv_block'):
    """
    Convolutional block with option for quantization
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size
        strides: Stride value
        use_quantization: Whether to use QKeras quantized layers
        precision: Quantization precision ('8bit', '16bit', '32bit')
        name: Block name
    """
    if use_quantization:
        # Select quantization config based on precision
        if precision == '8bit':
            qbits_config = quantized_bits_8_config
            qrelu_config = quantized_relu_config
        elif precision == '16bit':
            qbits_config = quantized_bits_16_config
            qrelu_config = quantized_relu_config
        elif precision == '32bit':
            qbits_config = quantized_bits_32_config
            qrelu_config = quantized_relu_config
        else:
            # Default to small config for unknown precision
            qbits_config = quantized_bits_8_config
            qrelu_config = quantized_relu_config
        
        # Use QKeras quantized layers
        x = QConv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_quantizer=quantized_bits(**qbits_config),
                   bias_quantizer=quantized_bits(**qbits_config),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                   name=f'{name}_qconv')(x)
        x = QBatchNormalization(
            gamma_quantizer=quantized_bits(**qbits_config),
            beta_quantizer=quantized_bits(**qbits_config),
            name=f'{name}_qbn')(x)
        x = QActivation(quantized_relu(**qrelu_config), name=f'{name}_qact')(x)
    else:
        # Use standard float32 layers
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                  name=f'{name}_conv')(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = Activation('relu', name=f'{name}_act')(x)
    
    return x

def attention_block(x, use_quantization=False, precision='16bit', name='attention'):
    """
    Channel attention block with option for quantization
    
    Args:
        x: Input tensor
        use_quantization: Whether to use QKeras quantized layers
        precision: Quantization precision ('8bit', '16bit', '32bit')
        name: Block name
    """
    channels = x.shape[-1]
    
    # Global average pooling
    y = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    y = tf.keras.layers.Reshape((1, 1, channels), name=f'{name}_reshape')(y)
    
    # Attention mechanism
    if use_quantization:
        # Select quantization config based on precision and layer characteristics
        if precision == '8bit':
            # Attention blocks in 8-bit layers use small config
            qbits_config = quantized_bits_8_config
            qrelu_config = quantized_relu_config
        elif precision == '16bit':
            qbits_config = quantized_bits_16_config
            qrelu_config = quantized_relu_config
        elif precision == '32bit':
            qbits_config = quantized_bits_32_config
            qrelu_config = quantized_relu_config
        else:
            qbits_config = quantized_bits_16_config
            qrelu_config = quantized_relu_config
        
        y = QConv2D(channels // 4, kernel_size=1, padding='same',
                   kernel_quantizer=quantized_bits(**qbits_config),
                   bias_quantizer=quantized_bits(**qbits_config),
                   name=f'{name}_qconv1')(y)
        y = QActivation(quantized_relu(**qrelu_config), name=f'{name}_qact1')(y)
        y = QConv2D(channels, kernel_size=1, padding='same',
                   kernel_quantizer=quantized_bits(**qbits_config),
                   bias_quantizer=quantized_bits(**qbits_config),
                   name=f'{name}_qconv2')(y)
        y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    else:
        y = Conv2D(channels // 4, kernel_size=1, padding='same',
                  name=f'{name}_conv1')(y)
        y = Activation('relu', name=f'{name}_act1')(y)
        y = Conv2D(channels, kernel_size=1, padding='same',
                  name=f'{name}_conv2')(y)
        y = Activation('sigmoid', name=f'{name}_sigmoid')(y)
    
    return Multiply(name=f'{name}_scale')([x, y])

def cbam_block(x, reduction_ratio=8, use_quantization=False, precision='16bit', name='cbam'):
    """
    Enhanced attention with both channel and spatial attention (CBAM) - HLS4ML optimized
    
    Args:
        x: Input tensor
        reduction_ratio: Reduction ratio for channel attention MLP
        use_quantization: Whether to use QKeras quantized layers
        precision: Quantization precision ('8bit', '16bit', '32bit')
        name: Block name
    """
    channels = x.shape[-1]
    
    # Channel attention - using only supported layers
    channel_avg = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_channel_avg')(x)
    channel_max = tf.keras.layers.GlobalMaxPooling2D(name=f'{name}_channel_max')(x)
    
    # Shared MLP for channel attention
    if use_quantization:
        # Select quantization config based on precision
        if precision == '8bit':
            qbits_config = quantized_bits_8_config
            qrelu_config = quantized_relu_config
        elif precision == '16bit':
            qbits_config = quantized_bits_16_config
            qrelu_config = quantized_relu_config
        elif precision == '32bit':
            qbits_config = quantized_bits_32_config
            qrelu_config = quantized_relu_config
        else:
            qbits_config = quantized_bits_16_config
            qrelu_config = quantized_relu_config
        
        # Channel attention MLP for average pooling
        channel_avg = QDense(channels // reduction_ratio, 
                           kernel_quantizer=quantized_bits(**qbits_config),
                           name=f'{name}_channel_mlp_avg')(channel_avg)
        channel_avg = QActivation(quantized_relu(**qrelu_config))(channel_avg)
        channel_avg = QDense(channels, 
                           kernel_quantizer=quantized_bits(**qbits_config),
                           name=f'{name}_channel_mlp_avg2')(channel_avg)
        
        # Channel attention MLP for max pooling
        channel_max = QDense(channels // reduction_ratio, 
                           kernel_quantizer=quantized_bits(**qbits_config),
                           name=f'{name}_channel_mlp_max')(channel_max)
        channel_max = QActivation(quantized_relu(**qrelu_config))(channel_max)
        channel_max = QDense(channels, 
                           kernel_quantizer=quantized_bits(**qbits_config),
                           name=f'{name}_channel_mlp_max2')(channel_max)
    else:
        # Standard float32 layers for channel attention
        channel_avg = Dense(channels // reduction_ratio, activation='relu', 
                          name=f'{name}_channel_mlp_avg')(channel_avg)
        channel_avg = Dense(channels, name=f'{name}_channel_mlp_avg2')(channel_avg)
        
        channel_max = Dense(channels // reduction_ratio, activation='relu', 
                          name=f'{name}_channel_mlp_max')(channel_max)
        channel_max = Dense(channels, name=f'{name}_channel_mlp_max2')(channel_max)
    
    # Combine channel attention
    channel_add = tf.keras.layers.Add()([channel_avg, channel_max])
    channel_attention = Activation('sigmoid')(channel_add)
    channel_attention = tf.keras.layers.Reshape((1, 1, channels))(channel_attention)
    x_channel = Multiply()([x, channel_attention])
    
    # Spatial attention - HLS4ML optimized (no Lambda layers)
    # Use Conv2D with 1x1 kernels to compute spatial attention
    # This replaces the Lambda layers with supported operations
    
    # Create spatial attention using Conv2D operations - FPGA optimized with smaller kernels
    # First, create a 1x1 conv to reduce channels to 2 for avg/max
    if use_quantization:
        spatial_reduce = QConv2D(2, kernel_size=1, padding='same',
                               kernel_quantizer=quantized_bits(**qbits_config),
                               name=f'{name}_spatial_reduce')(x_channel)
        # Use 3x3 kernel instead of 7x7 for FPGA compatibility
        spatial_attention = QConv2D(1, kernel_size=3, padding='same',
                                  kernel_quantizer=quantized_bits(**qbits_config),
                                  name=f'{name}_spatial_conv')(spatial_reduce)
        spatial_attention = QActivation('sigmoid')(spatial_attention)
    else:
        spatial_reduce = Conv2D(2, kernel_size=1, padding='same', 
                              name=f'{name}_spatial_reduce')(x_channel)
        # Use 3x3 kernel instead of 7x7 for FPGA compatibility
        spatial_attention = Conv2D(1, kernel_size=3, padding='same', 
                                 activation='sigmoid', name=f'{name}_spatial_conv')(spatial_reduce)
    
    return Multiply()([x_channel, spatial_attention])

def feature_pyramid_fusion(skip_connections, bridge_output, use_quantization=False, precision='32bit'):
    """
    Combine features from multiple scales using pyramid fusion - HLS4ML optimized
    
    Args:
        skip_connections: List of skip connection features from encoder
        bridge_output: Output from bridge block
        use_quantization: Whether to use QKeras quantized layers
        precision: Quantization precision for fusion layers
    """
    # Process bridge output first
    bridge_processed = conv_block(bridge_output, bridge_output.shape[-1], 
                                use_quantization=use_quantization, precision=precision, 
                                name='pyramid_bridge')
    
    # Start with bridge features
    fused_features = [bridge_processed]
    
    # Process each skip connection - ensure all features match bridge spatial dimensions
    for i, skip in enumerate(skip_connections):
        # Standardize all features to have the same number of channels as bridge
        target_channels = bridge_output.shape[-1]
        
        if skip.shape[-1] != target_channels:
            skip = conv_block(skip, target_channels, kernel_size=1, 
                            use_quantization=use_quantization, precision=precision, 
                            name=f'pyramid_conv_{i}')
        
        # Resize all features to match bridge spatial dimensions
        if skip.shape[1] != bridge_output.shape[1] or skip.shape[2] != bridge_output.shape[2]:
            # Use resize to match exact bridge dimensions
            skip = tf.keras.layers.Resizing(
                height=bridge_output.shape[1],
                width=bridge_output.shape[2],
                interpolation='bilinear'
            )(skip)
        
        fused_features.append(skip)
    
    return tf.keras.layers.Concatenate()(fused_features)

def create_segmentation_model(use_quantization=False):
    """
    Create semantic segmentation model
    
    Args:
        use_quantization: Whether to use QKeras quantized layers
    """
    tf.keras.backend.clear_session()
    
    # Input
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    
    # Initial feature extraction (INPUT block - 8-bit precision)
    x = conv_block(x, MODEL_CONFIG['init_filters'], use_quantization=use_quantization, precision='8bit', name='init_conv')
    
    # Encoder path with skip connections
    skip_connections = []
    
    # Encoder block 1 (ENCODER1 block - 8-bit precision)
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], strides=2, use_quantization=use_quantization, precision='8bit', name='enc1_down')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, precision='8bit', name='enc1_conv')
    skip_connections.append(x)  # Store features BEFORE dropout
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.1, name='dropout_encoder1')(x)
    
    # Encoder block 2 (ENCODER2 block - 16-bit precision)
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], strides=2, use_quantization=use_quantization, precision='16bit', name='enc2_down')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, precision='16bit', name='enc2_conv')
    x = cbam_block(x, use_quantization=use_quantization, precision='16bit', name='enc2_cbam')
    skip_connections.append(x)  # Store features BEFORE dropout
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.15, name='dropout_encoder2')(x)
    
    # Encoder block 3 (ENCODER3 block - 16-bit precision)
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], strides=2, use_quantization=use_quantization, precision='16bit', name='enc3_down')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, precision='16bit', name='enc3_conv')
    x = cbam_block(x, use_quantization=use_quantization, precision='16bit', name='enc3_cbam')
    skip_connections.append(x)  # Store features BEFORE dropout
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.2, name='dropout_encoder3')(x)
    
    # Bridge block (BRIDGE block - 32-bit precision for maximum accuracy)
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], strides=2, use_quantization=use_quantization, precision='32bit', name='bridge_down')
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], use_quantization=use_quantization, precision='32bit', name='bridge_conv1')
    x = conv_block(x, MODEL_CONFIG['stage4_filters'], use_quantization=use_quantization, precision='32bit', name='bridge_conv2')
    x = cbam_block(x, use_quantization=use_quantization, precision='32bit', name='bridge_cbam')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.25, name='dropout_bridge')(x)
    
    # Feature Pyramid Fusion - combine multi-scale features
    x = feature_pyramid_fusion(skip_connections, x, use_quantization=use_quantization, precision='32bit')
    
    # Decoder path
    # Decoder block 3 (DECODER3 block - 32-bit precision for critical features)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec3_up')(x)
    x = tf.keras.layers.Concatenate(name='dec3_concat')([x, skip_connections[2]])
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, precision='32bit', name='dec3_conv1')
    x = conv_block(x, MODEL_CONFIG['stage3_filters'], use_quantization=use_quantization, precision='32bit', name='dec3_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.2, name='dropout_decoder3')(x)
    
    # Decoder block 2 (DECODER2 block - 16-bit precision)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec2_up')(x)
    x = tf.keras.layers.Concatenate(name='dec2_concat')([x, skip_connections[1]])
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, precision='16bit', name='dec2_conv1')
    x = conv_block(x, MODEL_CONFIG['stage2_filters'], use_quantization=use_quantization, precision='16bit', name='dec2_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.15, name='dropout_decoder2')(x)
    
    # Decoder block 1 (DECODER1 block - 16-bit precision)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='dec1_up')(x)
    x = tf.keras.layers.Concatenate(name='dec1_concat')([x, skip_connections[0]])
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, precision='16bit', name='dec1_conv1')
    x = conv_block(x, MODEL_CONFIG['stage1_filters'], use_quantization=use_quantization, precision='16bit', name='dec1_conv2')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.1, name='dropout_decoder1')(x)
    
    # Final upsampling to original resolution
    x = tf.keras.layers.UpSampling2D(size=(2, 2), name='final_up')(x)
    
    # Final convolution before output (DECODER0 block - 8-bit precision)
    x = conv_block(x, MODEL_CONFIG['init_filters'], use_quantization=use_quantization, precision='8bit', name='final_conv')
    if USE_DROPOUT_DURING_TRAINING:
        x = Dropout(0.05, name='dropout_final')(x)
    
    # Output layer (OUTPUT block - 8-bit precision)
    if use_quantization:
        outputs = QConv2D(NUM_CLASSES, kernel_size=1, padding='same',
                         kernel_quantizer=quantized_bits(**quantized_bits_32_config),
                         bias_quantizer=quantized_bits(**quantized_bits_32_config),
                         activation='softmax', name='output_qconv')(x)
    else:
        outputs = Conv2D(NUM_CLASSES, kernel_size=1, padding='same',
                        activation='softmax', name='output_conv')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='segmentation_model')
    return model

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

def create_data_generators():
    """Create training and validation data generators"""
    train_image_dir = os.path.join(DATASET_DIR, 'leftImg8bit', 'train')
    val_image_dir = os.path.join(DATASET_DIR, 'leftImg8bit', 'val')
    train_mask_dir = os.path.join(DATASET_DIR, 'gtFine', 'train')
    val_mask_dir = os.path.join(DATASET_DIR, 'gtFine', 'val')
    
    train_generator = CityscapesDataGenerator(
        train_image_dir, train_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=True,
        augment=True
    )
    
    val_generator = CityscapesDataGenerator(
        val_image_dir, val_mask_dir,
        batch_size=BATCH_SIZE,
        subset_ratio=SUBSET_RATIO,
        target_height=INPUT_HEIGHT,
        target_width=INPUT_WIDTH,
        shuffle=False,
        augment=False
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
        if layer_name.endswith('_qconv'):
            float32_name = layer_name.replace('_qconv', '_conv')
        elif layer_name.endswith('_qbn'):
            float32_name = layer_name.replace('_qbn', '_bn')
        else:
            float32_name = layer_name
        
        if float32_name in float32_layers:
            float32_layer = float32_layers[float32_name]
            
            try:
                # Transfer weights for compatible layers
                if hasattr(qat_layer, 'set_weights') and hasattr(float32_layer, 'get_weights'):
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
        try:
            if TRAIN_QAT and not LOAD_MODEL_PATH.endswith('_qat'):
                # Load float32 model and convert to QAT
                print("Loading float32 model and converting to QAT...")
                try:
                    float32_model = tf.keras.models.load_model(LOAD_MODEL_PATH, compile=False)
                except Exception as e:
                    print(f"Failed to load full model, trying weights-only: {e}")
                    # Create model and load weights
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
                    # Create model and load weights
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
                print(f"Failed to load full checkpoint, trying weights-only: {e}")
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

    # Compile model with appropriate learning rate
    if USER_LR is not None:
        learning_rate = USER_LR
        print(f"Using user-specified learning rate: {learning_rate}")
    elif TRAIN_QAT:
        learning_rate = 1e-7
        print(f"Using recommended QAT learning rate: {learning_rate}")
    else:
        learning_rate = 7e-4
        print(f"Using recommended float32 learning rate: {learning_rate}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=weighted_combined_loss,  # Use weighted combined loss (best for scattered pixels)
        metrics=[
            "accuracy",
            dice_loss,
            mean_iou,
            cars,
            road,
            sidewalk,
            traffic_light,
            traffic_sign
            # precision_metric,
            # recall_metric,
            # f1_score_metric
        ],
    )
    print(f"Model compiled with learning rate: {learning_rate}")

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
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(viz_dir, exist_ok=True)

    # Enhanced callbacks with proper state management
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_checkpoint_epoch_{{epoch:02d}}.keras'),
            save_weights_only=True,  # Save weights only for compatibility
            save_best_only=False,
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{model_prefix}_best.keras'),
            save_weights_only=True,  # Save weights only for compatibility
            save_best_only=True,
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            factor=0.5 if TRAIN_QAT else 0.7,  # More aggressive reduction for car performance
            patience=12 if TRAIN_QAT else 8,   # Less patience to react faster
            verbose=1,
            min_lr=1e-8 if TRAIN_QAT else 5e-6   # Lower minimum learning rate
        ),
        EarlyStopping(
            monitor='val_mean_iou',  # Monitor IoU instead of loss
            mode='max',  # Higher IoU is better
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        PredictionVisualizationCallback(
            val_generator=val_generator,
            save_dir=viz_dir,
            num_samples=4,
            save_frequency=5
        ),
        DetailedMetricsCallback(
            val_generator=val_generator,
            save_dir=metrics_dir,
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

        # Save final model (compatible format)
        final_model_path = os.path.join(model_dir, f'{model_prefix}_final.keras')
        try:
            model.save(final_model_path, save_format='keras')
            print(f"Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"Warning: Could not save full model, saving weights only: {e}")
            model.save_weights(final_model_path)
            print(f"Final weights saved to: {final_model_path}")

        # Create and save clean model for HLS4ML export (without dropout)
        print("Creating clean model for HLS4ML export...")
        clean_model = create_hls4ml_export_model(trained_model=model, use_quantization=TRAIN_QAT)
        clean_model_path = os.path.join(model_dir, f'{model_prefix}_clean_for_hls4ml.keras')
        try:
            clean_model.save(clean_model_path, save_format='keras')
            print(f"Clean model (no dropout) saved to: {clean_model_path}")
        except Exception as e:
            print(f"Warning: Could not save full clean model, saving weights only: {e}")
            clean_model.save_weights(clean_model_path)
            print(f"Clean weights saved to: {clean_model_path}")

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
    print("HLS4ML Semantic Segmentation Training")
    print("="*60)
    
    if not TRAIN_QAT:
        print("Phase 1: Training float32 baseline model")
        print("This will create a high-quality baseline model without quantization")
        print("After training completes, use --trainQAT=True to apply QAT")
    else:
        print("Phase 2: Applying Quantization-Aware Training (QAT)")
        print("This will fine-tune the model with quantized layers for FPGA deployment")
        if LOAD_MODEL_PATH:
            print(f"Loading baseline model from: {LOAD_MODEL_PATH}")
        else:
            print("Training QAT model from scratch (not recommended)")
    
    print("="*60)
    
    train_model()
