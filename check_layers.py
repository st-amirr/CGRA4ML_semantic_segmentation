import tensorflow as tf
from train import create_segmentation_model

print("Checking actual layer names in the model...")

# Create model
model = create_segmentation_model()
print(f"Model has {len(model.layers)} layers")

# Check all layer names
print("\nAll layer names:")
for i, layer in enumerate(model.layers):
    print(f"{i:3d}: {layer.name} ({type(layer).__name__})")

# Check for quantization-related layers
print("\nQuantization-related layers:")
quantized_layers = []
for layer in model.layers:
    layer_name = layer.name.lower()
    layer_type = type(layer).__name__.lower()
    
    # Check various quantization patterns
    if any(qtype in layer_name for qtype in ['qconv', 'qactivation', 'qbatch', 'q_', 'quant']):
        quantized_layers.append((layer.name, type(layer).__name__))
    elif any(qtype in layer_type for qtype in ['qconv', 'qactivation', 'qbatch', 'quant']):
        quantized_layers.append((layer.name, type(layer).__name__))

if quantized_layers:
    print(f"Found {len(quantized_layers)} quantization layers:")
    for name, layer_type in quantized_layers:
        print(f"  {name} ({layer_type})")
else:
    print("No quantization layers found with current search patterns")
    
    # Try broader search
    print("\nTrying broader search...")
    for layer in model.layers:
        if 'q' in layer.name.lower() or 'quant' in str(type(layer)).lower():
            print(f"  {layer.name} ({type(layer).__name__})")

print("\nModel summary:")
model.summary(show_trainable=True) 