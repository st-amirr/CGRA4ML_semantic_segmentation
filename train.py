import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=20000)]
            )
            
        # Disable layout optimizer and other problematic optimizations
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': False,  # Disable layout optimization
            'constant_folding': True,
            'shape_optimization': False,  # Disable shape optimization
            'remapping': False,  # Disable remapping
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'scoped_allocator_optimization': False,  # Disable scoped allocator
            'implementation_selector': False,  # Disable implementation selector
        })
        
        print("GPU memory growth enabled with optimizations configured for QAT")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")

# Configure TensorFlow to use CPU efficiently
n_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(n_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_cores)

# Enable memory growth and tensor allocation reporting
tf.config.set_soft_device_placement(True)

# Disable eager execution for better memory tracking
tf.compat.v1.disable_eager_execution()

# Add TF memory logger
tf.debugging.set_log_device_placement(False)  # Set to True only for debugging

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization
from qkeras import QActivation, QConv2D, QConv2DTranspose, QBatchNormalization, quantized_bits, quantized_relu
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from PIL import Image
import cv2
from pathlib import Path

# Define constants
INPUT_HEIGHT = 1024
INPUT_WIDTH = 2048
BATCH_SIZE = 1  # Keep at 1 for memory constraints
EPOCHS = 50
NUM_CLASSES = 34
DATASET_DIR = '/cityscapes'
GRADIENT_ACCUMULATION_STEPS = 4  # Keep at 1 for now
NUM_WORKERS = 1  # Reduced for stability
PREFETCH_SIZE = 4  # Reduced for stability

# Define quantization parameters optimized for FPGA implementation
quantized_bits_param = {
    'bits': 8,
    'integer': 4,
    'symmetric': True,  # For better HLS4ML compatibility
    'alpha': 1.0,      # Keep weights in [-8,+8] range
    'use_stochastic_rounding': True,  # Enable for better training stability
    'scale_axis': None
}

quantized_relu_param = {
    'bits': 8,
    'integer': 4,
    'use_stochastic_rounding': True  # Enable for better training stability
}

def conv_block(x, filters, kernel_size=3, strides=1, name=None):
    """FPGA-optimized convolution block with QAT"""
    x = QConv2D(filters, kernel_size, strides=strides, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                kernel_initializer='glorot_uniform',  # Explicit initializer for reproducibility
                name=f'{name}_conv')(x)
    x = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param),
        beta_quantizer=quantized_bits(**quantized_bits_param),
        mean_quantizer=quantized_bits(**quantized_bits_param),
        variance_quantizer=quantized_bits(**quantized_bits_param),
        momentum=0.9,
        epsilon=1e-5,
        name=f'{name}_bn'
    )(x)
    x = QActivation(quantized_relu(**quantized_relu_param), name=f'{name}_act')(x)
    return x

def c2f_block(x, filters, n=1, name=None):
    """YOLOv8 C2f block with quantization optimized for VCK5000"""
    # First conv
    x = conv_block(x, filters, name=f'{name}_conv1')
    
    # Split
    y = x
    
    # Bottleneck blocks - optimized for VCK5000
    for i in range(n):
        # Use 1x1 conv for channel reduction
        y = conv_block(y, filters//2, kernel_size=1, name=f'{name}_bottleneck{i}_conv1')
        y = conv_block(y, filters, kernel_size=3, name=f'{name}_bottleneck{i}_conv2')
        x = tf.keras.layers.Concatenate(name=f'{name}_concat{i}')([x, y])
    
    return x

# Add SPPF module for better multi-scale feature extraction
def sppf_block(x, channels, name):
    """Spatial Pyramid Pooling - Fast (SPPF) with quantization"""
    y1 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5')(x)
    y2 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5_2')(y1)
    y3 = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=f'{name}_pool5_3')(y2)
    
    concat = tf.keras.layers.Concatenate(name=f'{name}_concat')([x, y1, y2, y3])
    
    return conv_block(concat, channels, kernel_size=1, name=f'{name}_conv')

# Efficient Channel Attention Module
def channel_attention(x, ratio=16, name=''):
    """Lightweight Squeeze-and-Excitation block with quantization"""
    channels = x.shape[-1]
    
    # Global average pooling
    y = tf.keras.layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    y = tf.keras.layers.Reshape((1, 1, channels), name=f'{name}_reshape')(y)
    
    # Single conv for efficiency
    y = QConv2D(channels, 1,
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name=f'{name}_attention')(y)
    y = QActivation('sigmoid', name=f'{name}_sigmoid')(y)
    
    return tf.keras.layers.Multiply(name=f'{name}_scale')([x, y])

def create_segmentation_model():
    # Clear any existing models and reset state
    tf.keras.backend.clear_session()
    
    inputs = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='input')
    x = inputs
    
    # Initial conv
    x = conv_block(x, 32, name='init_conv')
    x = QConv2D(32, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name='downsample1')(x)  # 512x1024
    
    # Stage 1 - 1/4 resolution
    x = c2f_block(x, 64, n=2, name='stage1')
    skip1 = x
    x = QConv2D(64, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name='downsample2')(x)  # 256x512
    
    # Stage 2 - 1/8 resolution
    x = c2f_block(x, 128, n=2, name='stage2')
    skip2 = x
    x = QConv2D(128, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name='downsample3')(x)  # 128x256
    
    # Stage 3 - 1/16 resolution
    x = c2f_block(x, 256, n=1, name='stage3')
    x = channel_attention(x, name='ca_stage3')  # Add attention only at deeper layers
    skip3 = x
    x = QConv2D(256, kernel_size=3, strides=2, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name='downsample4')(x)  # 64x128
    
    # Stage 4 - 1/32 resolution (bottleneck)
    x = c2f_block(x, 512, n=1, name='stage4')
    x = channel_attention(x, name='ca_stage4')  # Add attention at bottleneck
    
    # Add lightweight SPPF for multi-scale feature extraction
    x = sppf_block(x, 512, name='sppf')
    
    # Up 1 - 1/16 resolution
    x = c2f_block(x, 512, n=1, name='pre_up1')
    x = QConv2DTranspose(256, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up1_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat1')([x, skip3])
    x = conv_block(x, 256, name='up1_conv')
    x = c2f_block(x, 256, n=1, name='up1_c2f')
    x = channel_attention(x, name='ca_up1')  # Keep attention for important upsampling stages
    
    # Up 2 - 1/8 resolution
    x = c2f_block(x, 256, n=1, name='pre_up2')
    x = QConv2DTranspose(128, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up2_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat2')([x, skip2])
    x = conv_block(x, 128, name='up2_conv')
    x = c2f_block(x, 128, n=1, name='up2_c2f')
    
    # Up 3 - 1/4 resolution
    x = c2f_block(x, 128, n=1, name='pre_up3')
    x = QConv2DTranspose(64, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up3_transpose')(x)
    x = tf.keras.layers.Concatenate(name='concat3')([x, skip1])
    x = conv_block(x, 64, name='up3_conv')
    x = c2f_block(x, 64, n=1, name='up3_c2f')
    
    # Up 4 - Original resolution
    x = c2f_block(x, 64, n=1, name='pre_up4')
    x = QConv2DTranspose(32, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up4_transpose')(x)
    x = conv_block(x, 32, name='up4_conv')
    x = c2f_block(x, 32, n=1, name='up4_c2f')
    
    # Output layer with quantized softmax
    outputs = QConv2D(NUM_CLASSES, (1, 1),
                     kernel_quantizer=quantized_bits(**quantized_bits_param),
                     bias_quantizer=quantized_bits(**quantized_bits_param),
                     kernel_initializer='glorot_uniform',
                     name='output')(x)
    outputs = QActivation(quantized_bits(**quantized_bits_param), name='output_quant')(outputs)
    outputs = tf.keras.layers.Activation('softmax', name='output_softmax')(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class CityscapesDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        
        # Pre-compute image sizes for faster resizing
        self.target_size = (INPUT_WIDTH, INPUT_HEIGHT)
        print(f"\n[DEBUG] DataGenerator target size: {self.target_size}")
        
        # Initialize augmentation if needed
        self.aug = None
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        
        # Pre-allocate memory for batch
        self.X = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
        self.Y = np.zeros((self.batch_size, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
        
        print(f"Initialized data generator with {len(self.image_paths)} images")
        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        gc.collect()
        tf.keras.backend.clear_session()
    
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
            
            return img, mask_one_hot
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.image_paths[i] for i in batch_indices]
        batch_y = [self.mask_paths[i] for i in batch_indices]
        
        # Use pre-allocated arrays
        for i, (img_path, mask_path) in enumerate(zip(batch_x, batch_y)):
            img, mask = self.load_and_preprocess_image(img_path, mask_path)
            if img is not None and mask is not None:
                self.X[i] = img
                self.Y[i] = mask
        
        return self.X, self.Y

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Clean up memory at the end of each epoch"""
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Log memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"\nMemory usage after epoch {epoch}: {memory_usage:.2f} MB")

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
        
        # Update state variables using scatter_nd_update
        indices = tf.range(self.num_classes, dtype=tf.int32)
        indices = tf.reshape(indices, [-1, 1])
        
        self.intersection.assign_add(tf.scatter_nd(indices, tf.stack(intersections), [self.num_classes]))
        self.union.assign_add(tf.scatter_nd(indices, tf.stack(unions), [self.num_classes]))

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
    # Create models directory if it doesn't exist
    os.makedirs('/app/models', exist_ok=True)
    
    # Find the latest saved model
    model_files = [f for f in os.listdir('/app/models') if f.endswith('.h5')]
    initial_epoch = 0
    
    if model_files:
        # Sort by modification time to get the latest
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('/app/models', x)))
        print(f"\nFound latest model: {latest_model}")
        
        # Extract epoch number if it exists in filename
        if 'epoch_' in latest_model:
            try:
                initial_epoch = int(latest_model.split('epoch_')[1].split('_')[0])
                print(f"Resuming from epoch {initial_epoch}")
            except:
                print("Could not determine epoch number from filename")
        
        # Load the complete model
        try:
            print(f"Loading model from {latest_model}")
            model = tf.keras.models.load_model('/app/models/' + latest_model,
                                           custom_objects={
                                               'QConv2D': QConv2D,
                                               'QBatchNormalization': QBatchNormalization,
                                               'QActivation': QActivation,
                                               'IoUMetric': IoUMetric,
                                               'quantized_bits': quantized_bits,
                                               'quantized_relu': quantized_relu
                                           })
            print("Successfully loaded model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Will start training from scratch")
            model = create_segmentation_model()
            initial_epoch = 0
    else:
        print("\nNo existing models found. Creating new model.")
        model = create_segmentation_model()
        initial_epoch = 0
    
    # Compile with QAT-specific settings and disable XLA
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=1e-04,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[IoUMetric()],
        run_eagerly=False,
        # Remove jit_compile if it was added
    )
    
    # Setup data paths
    cityscapes_root = '/data'
    print(f"\nLooking for Cityscapes dataset in: {cityscapes_root}")
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    print(f"\nFound {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create data generators with memory optimization
    train_gen = CityscapesDataGenerator(train_images, train_masks, BATCH_SIZE, augment=True)
    val_gen = CityscapesDataGenerator(val_images, val_masks, BATCH_SIZE, augment=False)
    
    # Add HLS4ML model saving callback
    class HLS4MLModelSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
                # Save quantized weights in HLS4ML compatible format
                model_save_quantized_weights(
                    self.model,
                    f'/app/models/hls4ml_weights_epoch_{epoch+1}.h5'
                )
    
    callbacks = [
        ModelCheckpoint(
            '/app/models/best_model.h5',
            save_best_only=True,
            monitor='val_iou',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        ModelCheckpoint(
            '/app/models/checkpoint_epoch_{epoch:02d}.h5',
            save_freq='epoch',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            patience=10,
            monitor='val_iou',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        MemoryCleanupCallback(),
        HLS4MLModelSaver()
    ]
    
    # Train with gradient accumulation
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            workers=NUM_WORKERS,
            use_multiprocessing=False,
            max_queue_size=PREFETCH_SIZE,
            steps_per_epoch=len(train_gen) // GRADIENT_ACCUMULATION_STEPS,
            initial_epoch=initial_epoch - 1 if initial_epoch > 0 else 0
        )
        
        # Save final model in both formats
        print("\nSaving final models...")
        model.save('/app/models/final_model.h5')
        model_save_quantized_weights(model, '/app/models/final_model_hls4ml.h5')
        print("Models saved as 'final_model.h5' and 'final_model_hls4ml.h5'")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

#24/743 [..............................] - ETA: 3:05:34 - loss: 2.8576 - iou: 0.3420