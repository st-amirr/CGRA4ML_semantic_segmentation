import tensorflow as tf
import time
import numpy as np
from train import create_segmentation_model

print("Comprehensive comparison of all training approaches...")

# Create model
model = create_segmentation_model()
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Input shape: {x.shape}")
print(f"Model layers: {len(model.layers)}")

# Test 1: GPU only (current approach)
print("\n1. GPU only (current):")
start = time.time()
_ = model(x)
gpu_time = time.time() - start
print(f"   Time: {gpu_time:.3f}s")

# Test 2: CPU only
print("\n2. CPU only:")
with tf.device('/CPU:0'):
    start = time.time()
    _ = model(x)
    cpu_time = time.time() - start
print(f"   Time: {cpu_time:.3f}s")

# Test 3: Hybrid (input on GPU, model on CPU)
print("\n3. Hybrid (input on GPU, model on CPU):")
with tf.device('/GPU:0'):
    x_gpu = tf.constant(x)
start = time.time()
with tf.device('/CPU:0'):
    _ = model(x_gpu)
hybrid_time = time.time() - start
print(f"   Time: {hybrid_time:.3f}s")

# Test 4: Automatic device placement (new approach)
print("\n4. Automatic device placement (new hybrid):")
tf.config.set_soft_device_placement(True)

class HybridModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

hybrid_model = HybridModel(model)
start = time.time()
_ = hybrid_model(x)
auto_hybrid_time = time.time() - start
print(f"   Time: {auto_hybrid_time:.3f}s")

# Summary
print("\n" + "="*60)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("="*60)
print(f"GPU only:              {gpu_time:.3f}s")
print(f"CPU only:              {cpu_time:.3f}s")
print(f"Hybrid (manual):       {hybrid_time:.3f}s")
print(f"Hybrid (automatic):    {auto_hybrid_time:.3f}s")

# Find best approach
times = [gpu_time, cpu_time, hybrid_time, auto_hybrid_time]
approaches = ["GPU only", "CPU only", "Hybrid (manual)", "Hybrid (automatic)"]
best_idx = times.index(min(times))
best_time = times[best_idx]

print(f"\nBEST APPROACH: {approaches[best_idx]} ({best_time:.3f}s)")

# Calculate improvements
print(f"\nPERFORMANCE IMPROVEMENTS:")
for i, (approach, time_val) in enumerate(zip(approaches, times)):
    if i != best_idx:
        improvement = ((time_val - best_time) / time_val) * 100
        print(f"  {approach}: {improvement:.1f}% slower than best")

print(f"\nRECOMMENDATION:")
if best_idx == 0:
    print("  Use GPU-only training (current approach)")
elif best_idx == 1:
    print("  Use CPU-only training for quantization")
elif best_idx == 2:
    print("  Use manual hybrid approach (input on GPU, model on CPU)")
else:
    print("  Use automatic hybrid approach (TensorFlow device placement)")

print("="*60) 