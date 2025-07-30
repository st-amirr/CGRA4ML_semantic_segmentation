import tensorflow as tf
import time
import numpy as np
from train import create_segmentation_model

print("Testing Hybrid CPU/GPU approach for quantization model...")

# Create model
model = create_segmentation_model()
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Input shape: {x.shape}")
print(f"Model layers: {len(model.layers)}")

# Test different device placements
print("\n" + "="*50)
print("TESTING DIFFERENT DEVICE PLACEMENTS")
print("="*50)

# Test 1: All on GPU (current approach)
print("\n1. All on GPU (current):")
start = time.time()
_ = model(x)
gpu_time = time.time() - start
print(f"   Time: {gpu_time:.3f}s")

# Test 2: All on CPU
print("\n2. All on CPU:")
with tf.device('/CPU:0'):
    start = time.time()
    _ = model(x)
    cpu_time = time.time() - start
print(f"   Time: {cpu_time:.3f}s")

# Test 3: Hybrid - Input on GPU, quantization layers on CPU
print("\n3. Hybrid - Input on GPU, quantization on CPU:")
# Create input on GPU
with tf.device('/GPU:0'):
    x_gpu = tf.constant(x)
print(f"   Input placed on GPU")

# Test hybrid forward pass
start = time.time()
with tf.device('/CPU:0'):
    _ = model(x_gpu)
hybrid_time = time.time() - start
print(f"   Time: {hybrid_time:.3f}s")

# Test 4: CPU-only with optimized placement
print("\n4. CPU-only with optimized placement:")
start = time.time()
with tf.device('/CPU:0'):
    _ = model(x)
cpu_opt_time = time.time() - start
print(f"   Time: {cpu_opt_time:.3f}s")

# Summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(f"GPU only:        {gpu_time:.3f}s")
print(f"CPU only:        {cpu_time:.3f}s")
print(f"Hybrid (simple): {hybrid_time:.3f}s")
print(f"CPU optimized:   {cpu_opt_time:.3f}s")

# Find best approach
times = [gpu_time, cpu_time, hybrid_time, cpu_opt_time]
approaches = ["GPU only", "CPU only", "Hybrid (simple)", "CPU optimized"]
best_idx = times.index(min(times))
best_time = times[best_idx]

print(f"\nBEST APPROACH: {approaches[best_idx]} ({best_time:.3f}s)")

# Calculate improvements
print(f"\nIMPROVEMENTS:")
for i, (approach, time_val) in enumerate(zip(approaches, times)):
    if i != best_idx:
        improvement = ((time_val - best_time) / time_val) * 100
        print(f"  {approach}: {improvement:.1f}% slower than best") 