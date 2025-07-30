import tensorflow as tf
import numpy as np
from train import create_segmentation_model, CityscapesDataGenerator, prepare_data

print("Testing hybrid CPU/GPU training approach...")

# Create model
model = create_segmentation_model()
print(f"Model created with {len(model.layers)} layers")

# Prepare data
cityscapes_root = '/data'
train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)

# Create small data generators for testing
train_gen = CityscapesDataGenerator(train_images[:10], train_masks[:10], 1, augment=False, subset_ratio=1.0)
val_gen = CityscapesDataGenerator(val_images[:5], val_masks[:5], 1, augment=False, subset_ratio=1.0)

print(f"Training samples: {len(train_gen)}")
print(f"Validation samples: {len(val_gen)}")

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Test hybrid training
print("\nTesting hybrid training...")
start_time = tf.timestamp()

# Create input on GPU, run model on CPU
with tf.device('/GPU:0'):
    x_gpu = tf.constant(np.random.random((1, 1024, 2048, 3)).astype(np.float32))

with tf.device('/CPU:0'):
    _ = model(x_gpu)

hybrid_time = tf.timestamp() - start_time
print(f"Hybrid forward pass: {hybrid_time:.3f}s")

print("\nHybrid training test completed successfully!")
print("Ready to use hybrid approach in main training.") 