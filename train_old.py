import os
import numpy as np
import tensorflow as tf
import multiprocessing
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Configure TensorFlow to use GPU memory efficiently
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        memory_limit = 20000  # 20GB in MB to utilize shared GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Limit GPU memory usage to prevent OOM
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
        # Disable layout optimizer to prevent NCHW-NHWC conversion errors
        tf.config.optimizer.set_experimental_options({
            "layout_optimizer": False
        })
        print(f"GPU memory growth enabled and limited to {memory_limit/1024:.1f}GB")
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

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization
from qkeras import *
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
    'symmetric': False,  # Changed to match previous training
    'alpha': 'auto_po2',  # Changed to match previous training
    'use_stochastic_rounding': False,
    'scale_axis': None
}

# Separate parameters for activations
quantized_relu_param = {
    'bits': 8,
    'integer': 4
}

def conv_block(x, filters, kernel_size=3, strides=1, name=None):
    """FPGA-optimized convolution block with QAT"""
    x = QConv2D(filters, kernel_size, strides=strides, padding='same',
                kernel_quantizer=quantized_bits(**quantized_bits_param),
                bias_quantizer=quantized_bits(**quantized_bits_param),
                name=f'{name}_conv')(x)
    x = QBatchNormalization(
        gamma_quantizer=quantized_bits(**quantized_bits_param),
        beta_quantizer=quantized_bits(**quantized_bits_param),
        mean_quantizer=quantized_bits(**quantized_bits_param),
        variance_quantizer=quantized_bits(**quantized_bits_param),
        momentum=0.9,  # Add momentum parameter - safe change
        epsilon=1e-5,  # Add epsilon - safe change
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

def create_segmentation_model():
    # Input layer with quantization
    inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
    x = QActivation(quantized_bits(**quantized_bits_param), name='input_quant')(inputs)
    
    # Initial stem - optimized for FPGA
    x = conv_block(x, 32, kernel_size=3, name='stem')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Store skip connections for decoder
    skip_connections = []
    
    # Backbone (YOLOv8-style) - optimized for FPGA
    # Stage 1 - 1/4 resolution
    x = c2f_block(x, 64, n=2, name='stage1')  # Increased filters and n
    skip_connections.append(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 2 - 1/8 resolution
    x = c2f_block(x, 128, n=2, name='stage2')  # Increased filters and n
    skip_connections.append(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 3 - 1/16 resolution
    x = c2f_block(x, 256, n=1, name='stage3')
    skip_connections.append(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 4 - 1/32 resolution
    x = c2f_block(x, 512, n=1, name='stage4')
    
    # Decoder (U-Net style with YOLOv8 blocks) - optimized for FPGA
    # Up 1 - 1/16 resolution
    x = QConv2DTranspose(256, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up1_transpose')(x)
    x = conv_block(x, 256, name='up1_conv')
    x = c2f_block(x, 256, n=1, name='up1_c2f')
    
    # Up 2 - 1/8 resolution
    x = QConv2DTranspose(128, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up2_transpose')(x)
    x = conv_block(x, 128, name='up2_conv')
    x = c2f_block(x, 128, n=1, name='up2_c2f')
    
    # Up 3 - 1/4 resolution
    x = QConv2DTranspose(64, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up3_transpose')(x)
    x = conv_block(x, 64, name='up3_conv')
    x = c2f_block(x, 64, n=1, name='up3_c2f')
    
    # Up 4 - 1/2 resolution
    x = QConv2DTranspose(32, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='up4_transpose')(x)
    x = conv_block(x, 32, name='up4_conv')
    x = c2f_block(x, 32, n=1, name='up4_c2f')
    
    # Final upsampling to full resolution
    x = QConv2DTranspose(32, kernel_size=2, strides=2, padding='same',
                        kernel_quantizer=quantized_bits(**quantized_bits_param),
                        bias_quantizer=quantized_bits(**quantized_bits_param),
                        name='final_transpose')(x)
    
    # Output layer with quantized softmax
    outputs = QConv2D(NUM_CLASSES, (1, 1),
                     kernel_quantizer=quantized_bits(**quantized_bits_param),
                     bias_quantizer=quantized_bits(**quantized_bits_param),
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
        
        # Initialize augmentation if needed
        self.aug = None
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
            ])
        
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

# Custom accuracy metric for semantic segmentation
class SegmentationAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='segmentation_accuracy', **kwargs):
        super(SegmentationAccuracy, self).__init__(name=name, **kwargs)
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')
        self.correct_pixels = self.add_weight(name='correct_pixels', initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices more efficiently
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Calculate matches using static shape when possible
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        
        # Update metrics efficiently using reduce_sum
        num_pixels = tf.cast(tf.size(matches), tf.float32)
        correct = tf.reduce_sum(matches)
        
        # Update state variables
        self.total_pixels.assign_add(num_pixels)
        self.correct_pixels.assign_add(correct)

    def result(self):
        # Avoid division by zero
        return tf.math.divide_no_nan(self.correct_pixels, self.total_pixels)

    def reset_state(self):
        self.total_pixels.assign(0.)
        self.correct_pixels.assign(0.)

    def get_config(self):
        base_config = super().get_config()
        return base_config

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
                                               'SegmentationAccuracy': SegmentationAccuracy,
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
    
    # Always compile with new learning rate
    print("Compiling model with specified learning rate...")
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
        metrics=[SegmentationAccuracy()],
        run_eagerly=False
    )
    
    # Setup data paths
    cityscapes_root = '/data'
    print(f"\nLooking for Cityscapes dataset in: {cityscapes_root}")
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    print(f"\nFound {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create data generators
    train_gen = CityscapesDataGenerator(train_images, train_masks, BATCH_SIZE, augment=True)
    val_gen = CityscapesDataGenerator(val_images, val_masks, BATCH_SIZE, augment=False)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            '/app/models/best_model.h5',
            save_best_only=True,
            monitor='val_segmentation_accuracy',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        ModelCheckpoint(
            '/app/models/checkpoint_epoch_{epoch:02d}_batch_{batch:04d}.h5',
            save_freq=50,
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            patience=10,
            monitor='val_segmentation_accuracy',
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
        MemoryCleanupCallback()
    ]
    
    # Train the model with gradient accumulation
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
            initial_epoch=initial_epoch - 1 if initial_epoch > 0 else 0  # Adjust for zero-based epoch counting
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('/app/models', exist_ok=True)
        
        # Save the final model
        print("\nSaving final model...")
        model.save('/app/models/final_model.h5')
        print("Model saved as '/app/models/final_model.h5'")
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save('/app/models/interrupted_model.h5')
        print("Model saved as '/app/models/interrupted_model.h5'")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

#24/743 [..............................] - ETA: 3:05:34 - loss: 2.8576 - segmentation_accuracy: 0.3420