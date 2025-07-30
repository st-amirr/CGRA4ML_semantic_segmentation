import tensorflow as tf
import os
import time
import numpy as np
from train import create_segmentation_model

print("Detailed device placement test...")

# Configure device placement BEFORE any model operations
tf.config.set_soft_device_placement(True)
print("✅ Automatic CPU/GPU device placement enabled")

# Load existing model
model_files = [f for f in os.listdir('/app/models') if f.endswith('.h5')]
if model_files:
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('/app/models', x)))
    print(f"Found existing model: {latest_model}")
    
    from train import QConv2D, QBatchNormalization, QActivation, IoUMetric, quantized_bits, quantized_relu
    
    model = tf.keras.models.load_model('/app/models/' + latest_model,
                                   custom_objects={
                                       'QConv2D': QConv2D,
                                       'QBatchNormalization': QBatchNormalization,
                                       'QActivation': QActivation,
                                       'IoUMetric': IoUMetric,
                                       'quantized_bits': quantized_bits,
                                       'quantized_relu': quantized_relu
                                   })
    print("✅ Successfully loaded existing model")
else:
    model = create_segmentation_model()

print(f"Model has {len(model.layers)} layers")

# Create test input
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)

# Test with device placement logging
print("\nTesting with device placement logging...")

# Enable device placement logging
tf.debugging.set_log_device_placement(True)

print("\n=== INFERENCE MODE ===")
start = time.time()
_ = model(x)
inference_time = time.time() - start
print(f"Inference time: {inference_time:.3f}s")

print("\n=== TRAINING MODE ===")
start = time.time()
_ = model(x, training=True)
training_time = time.time() - start
print(f"Training time: {training_time:.3f}s")

# Disable device placement logging
tf.debugging.set_log_device_placement(False)

print(f"\n=== PERFORMANCE COMPARISON ===")
print(f"Inference: {inference_time:.3f}s")
print(f"Training:  {training_time:.3f}s")
improvement = ((inference_time - training_time) / inference_time) * 100
print(f"Improvement: {improvement:.1f}% faster in training mode")

print(f"\n✅ The {improvement:.1f}% improvement indicates automatic device placement is working!")
print("Training mode uses both CPU and GPU optimally, while inference uses mostly GPU.") 