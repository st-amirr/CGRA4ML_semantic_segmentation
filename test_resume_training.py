import tensorflow as tf
import os
import time
import numpy as np
from train import create_segmentation_model

print("Testing device placement when resuming from checkpoint...")

# Configure device placement BEFORE any model operations
tf.config.set_soft_device_placement(True)
print("✅ Automatic CPU/GPU device placement enabled")

# Check if there are existing models
model_files = [f for f in os.listdir('/app/models') if f.endswith('.h5')]
if model_files:
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('/app/models', x)))
    print(f"Found existing model: {latest_model}")
    
    # Try to load the model
    try:
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = create_segmentation_model()
        print("✅ Created new model instead")
else:
    print("No existing models found, creating new model")
    model = create_segmentation_model()

print(f"Model has {len(model.layers)} layers")

# Test forward pass to see device placement in action
x = np.random.random((1, 1024, 2048, 3)).astype(np.float32)
print(f"Testing forward pass with input shape: {x.shape}")

# Test inference
print("\nTesting inference...")
start = time.time()
_ = model(x)
inference_time = time.time() - start
print(f"Inference time: {inference_time:.3f}s")

# Test training mode
print("\nTesting training mode...")
start = time.time()
_ = model(x, training=True)
training_time = time.time() - start
print(f"Training time: {training_time:.3f}s")

print("\n✅ Device placement test completed!")
print("The model should now use automatic CPU/GPU placement when resuming training.") 