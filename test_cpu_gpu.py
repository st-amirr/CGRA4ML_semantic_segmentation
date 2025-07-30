import tensorflow as tf
import time
import numpy as np
from train import create_segmentation_model

print("Testing CPU vs GPU for quantization model...")

# Create model
model = create_segmentation_model()
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Input shape: {x.shape}")
print(f"Model layers: {len(model.layers)}")

# Test GPU performance
print("\nTesting GPU performance...")
start = time.time()
_ = model(x)
gpu_time = time.time() - start
print(f"GPU forward pass: {gpu_time:.3f}s")

# Test CPU performance
print("\nTesting CPU performance...")
with tf.device('/CPU:0'):
    start = time.time()
    _ = model(x)
    cpu_time = time.time() - start
print(f"CPU forward pass: {cpu_time:.3f}s")

print(f"\nGPU vs CPU ratio: {gpu_time/cpu_time:.2f}x")
print(f"CPU is {cpu_time/gpu_time:.2f}x slower than GPU")

if cpu_time < gpu_time:
    print("CPU is FASTER for this quantization model!")
else:
    print("GPU is faster (as expected)") 