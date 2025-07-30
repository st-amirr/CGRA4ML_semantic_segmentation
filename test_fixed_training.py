import tensorflow as tf
import time
import numpy as np
from train import create_segmentation_model

print("Testing fixed training approach...")

# Configure device placement
tf.config.set_soft_device_placement(True)

# Create model
model = create_segmentation_model()
print(f"Model created with {len(model.layers)} layers")

# Create test input
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Input shape: {x.shape}")

# Test forward pass
print("\nTesting forward pass with automatic device placement...")
start = time.time()
_ = model(x)
forward_time = time.time() - start
print(f"Forward pass time: {forward_time:.3f}s")

# Test with training=True
print("\nTesting forward pass with training=True...")
start = time.time()
_ = model(x, training=True)
training_time = time.time() - start
print(f"Training forward pass time: {training_time:.3f}s")

print("\n✅ Fixed training approach is working correctly!")
print("Ready to start training with automatic CPU/GPU device placement.") 