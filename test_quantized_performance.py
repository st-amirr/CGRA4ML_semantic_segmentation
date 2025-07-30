import tensorflow as tf
import time
import numpy as np
from train import create_segmentation_model

print("Testing quantized model performance with different approaches...")

# Create quantized model
model = create_segmentation_model()
print(f"Model created with {len(model.layers)} layers")

# Check quantization layers
quantized_layers = []
for layer in model.layers:
    layer_type = type(layer).__name__
    if layer_type in ['QConv2D', 'QBatchNormalization', 'QActivation']:
        quantized_layers.append(layer.name)
print(f"Found {len(quantized_layers)} quantized layers")

# Create input that matches your training data
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Input shape: {x.shape}")

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

# Test 5: Multiple forward passes to warm up and get average
print("\n5. Multiple forward passes (warmup + average):")
warmup_passes = 3
test_passes = 5

# Warmup
print("   Warming up...")
for _ in range(warmup_passes):
    _ = hybrid_model(x)

# Test multiple passes
print("   Testing multiple passes...")
times = []
for i in range(test_passes):
    start = time.time()
    _ = hybrid_model(x)
    pass_time = time.time() - start
    times.append(pass_time)
    print(f"   Pass {i+1}: {pass_time:.3f}s")

avg_hybrid_time = np.mean(times)
std_hybrid_time = np.std(times)
print(f"   Average: {avg_hybrid_time:.3f}s ± {std_hybrid_time:.3f}s")

# Summary
print("\n" + "="*60)
print("QUANTIZED MODEL PERFORMANCE COMPARISON")
print("="*60)
print(f"GPU only:              {gpu_time:.3f}s")
print(f"CPU only:              {cpu_time:.3f}s")
print(f"Hybrid (manual):       {hybrid_time:.3f}s")
print(f"Hybrid (automatic):    {auto_hybrid_time:.3f}s")
print(f"Hybrid (avg, warmed):  {avg_hybrid_time:.3f}s ± {std_hybrid_time:.3f}s")

# Find best approach
times_list = [gpu_time, cpu_time, hybrid_time, auto_hybrid_time, avg_hybrid_time]
approaches = ["GPU only", "CPU only", "Hybrid (manual)", "Hybrid (automatic)", "Hybrid (warmed)"]
best_idx = times_list.index(min(times_list))
best_time = times_list[best_idx]

print(f"\nBEST APPROACH: {approaches[best_idx]} ({best_time:.3f}s)")

# Calculate improvements
print(f"\nPERFORMANCE IMPROVEMENTS:")
for i, (approach, time_val) in enumerate(zip(approaches, times_list)):
    if i != best_idx:
        improvement = ((time_val - best_time) / time_val) * 100
        print(f"  {approach}: {improvement:.1f}% slower than best")

print(f"\nQUANTIZATION IMPACT:")
print(f"  Model has {len(quantized_layers)} quantized layers")
print(f"  Quantization operations: {', '.join(quantized_layers[:5])}{'...' if len(quantized_layers) > 5 else ''}")

print("="*60) 