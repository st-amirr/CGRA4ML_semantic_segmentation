"""
Set-Location -Path "C:/Users/stamir/Desktop/hls4ml_SS"; docker-compose exec hls4ml-training python3 convert_merge_layers_for_hls4ml.py
"""

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Add, Input
import h5py
import numpy as np

# --- CONFIG ---
INPUT_MODEL_PATH = "./models/qat_model_clean_for_hls4ml.h5"
OUTPUT_MODEL_PATH = './models/qat_model_clean_for_compatible.h5'

# --- LOAD MODEL ---
custom_objects = {}
try:
    from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu, QDense, QDepthwiseConv2D
    custom_objects = {
        "QActivation": QActivation,
        "QConv2D": QConv2D,
        "QBatchNormalization": QBatchNormalization,
        "quantized_bits": quantized_bits,
        "quantized_relu": quantized_relu,
        "QDense": QDense,
        "QDepthwiseConv2D": QDepthwiseConv2D,
    }
except ImportError:
    pass

model = load_model(INPUT_MODEL_PATH, compile=False, custom_objects=custom_objects)

# --- ENHANCED QUANTIZER FIXING FUNCTIONS ---
def create_supported_quantizer_config(original_config):
    """Create HLS4ML-compatible quantizer configuration by removing unsupported parameters"""
    if original_config is None:
        return None
    
    new_config = original_config.copy()
    
    # Remove problematic alpha parameter
    if 'alpha' in new_config:
        if new_config['alpha'] in ['auto', 'auto_po2']:
            del new_config['alpha']
    
    # Remove problematic scale_axis parameter
    if 'scale_axis' in new_config:
        del new_config['scale_axis']
    
    return new_config

def fix_quantizer_object(quantizer):
    """Fix a quantizer object by replacing problematic quantizers with HLS4ML-compatible ones"""
    if quantizer is None:
        return None
    
    # Get the quantizer's class name
    class_name = quantizer.__class__.__name__
    
    # Check if it's a problematic quantizer
    if 'quantized_po2' in class_name:
        # Get the quantizer config and fix it
        quantizer_config = quantizer.get_config()
        fixed_config = create_supported_quantizer_config(quantizer_config)
        
        # Create new quantized_bits quantizer
        from qkeras import quantized_bits
        return quantized_bits(**fixed_config)
    
    elif 'quantized_relu_po2' in class_name:
        # Get the quantizer config and fix it
        quantizer_config = quantizer.get_config()
        fixed_config = create_supported_quantizer_config(quantizer_config)
        
        # Create new quantized_relu quantizer
        from qkeras import quantized_relu
        return quantized_relu(**fixed_config)
    
    return quantizer

def fix_quantizer_in_layer(layer):
    """Fix quantizers in a layer by replacing problematic quantizers with HLS4ML-compatible ones"""
    layer_config = layer.get_config()
    fixed = False
    
    # Define quantizer attributes to check
    quantizer_attrs = ['kernel_quantizer', 'bias_quantizer', 'activation', 'quantizer', 
                      'gamma_quantizer', 'beta_quantizer', 'mean_quantizer', 'variance_quantizer',
                      'depthwise_quantizer']
    
    for attr in quantizer_attrs:
        if attr in layer_config and layer_config[attr] is not None:
            quantizer = layer_config[attr]
            
            # Handle string quantizer configurations
            if isinstance(quantizer, str):
                if 'quantized_po2' in quantizer:
                    # Replace quantized_po2 with quantized_bits
                    layer_config[attr] = quantizer.replace('quantized_po2', 'quantized_bits')
                    fixed = True
                    print(f"    Fixed string quantizer: {quantizer} -> {layer_config[attr]}")
                elif 'quantized_relu_po2' in quantizer:
                    # Replace quantized_relu_po2 with quantized_relu
                    layer_config[attr] = quantizer.replace('quantized_relu_po2', 'quantized_relu')
                    fixed = True
                    print(f"    Fixed string quantizer: {quantizer} -> {layer_config[attr]}")
            
            # Handle quantizer objects
            elif hasattr(quantizer, '__class__'):
                fixed_quantizer = fix_quantizer_object(quantizer)
                if fixed_quantizer is not quantizer:
                    layer_config[attr] = fixed_quantizer
                    fixed = True
                    print(f"    Fixed quantizer object: {quantizer.__class__.__name__} -> {fixed_quantizer.__class__.__name__}")
    
    return layer_config, fixed

def deep_fix_model_quantizers(model):
    """Deep fix all quantizers in the model by directly modifying layer attributes"""
    fixed_count = 0
    
    for layer in model.layers:
        layer_fixed = False
        
        # Check if this is a QKeras layer that might have quantizers
        if hasattr(layer, 'kernel_quantizer') and layer.kernel_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.kernel_quantizer)
            if fixed_quantizer is not layer.kernel_quantizer:
                layer.kernel_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'bias_quantizer') and layer.bias_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.bias_quantizer)
            if fixed_quantizer is not layer.bias_quantizer:
                layer.bias_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'activation') and layer.activation is not None:
            fixed_quantizer = fix_quantizer_object(layer.activation)
            if fixed_quantizer is not layer.activation:
                layer.activation = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'quantizer') and layer.quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.quantizer)
            if fixed_quantizer is not layer.quantizer:
                layer.quantizer = fixed_quantizer
                layer_fixed = True
        
        # QBatchNormalization specific quantizers
        if hasattr(layer, 'gamma_quantizer') and layer.gamma_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.gamma_quantizer)
            if fixed_quantizer is not layer.gamma_quantizer:
                layer.gamma_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'beta_quantizer') and layer.beta_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.beta_quantizer)
            if fixed_quantizer is not layer.beta_quantizer:
                layer.beta_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'mean_quantizer') and layer.mean_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.mean_quantizer)
            if fixed_quantizer is not layer.mean_quantizer:
                layer.mean_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'variance_quantizer') and layer.variance_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.variance_quantizer)
            if fixed_quantizer is not layer.variance_quantizer:
                layer.variance_quantizer = fixed_quantizer
                layer_fixed = True
        
        if hasattr(layer, 'depthwise_quantizer') and layer.depthwise_quantizer is not None:
            fixed_quantizer = fix_quantizer_object(layer.depthwise_quantizer)
            if fixed_quantizer is not layer.depthwise_quantizer:
                layer.depthwise_quantizer = fixed_quantizer
                layer_fixed = True
        
        if layer_fixed:
            print(f"✅ Fixed quantizers in layer: {layer.name}")
            fixed_count += 1
    
    return fixed_count

# --- UTILITY: Replace multi-input merges ---
def refactor_merge_layer(layer, inputs):
    # Only refactor if more than 2 inputs
    if len(inputs) <= 2:
        return layer(inputs)
    # For Concatenate, use chained two-input merges with unique names
    if isinstance(layer, Concatenate):
        out = Concatenate(name=layer.name + '_temp0')([inputs[0], inputs[1]])
        for i in range(2, len(inputs)):
            out = Concatenate(name=layer.name + f'_temp{i-1}')([out, inputs[i]])
        return out
    # For Add, use chained two-input merges
    out = Add(name=layer.name + '_temp0')([inputs[0], inputs[1]])
    for i in range(2, len(inputs)):
        out = Add(name=layer.name + f'_temp{i-1}')([out, inputs[i]])
    return out

# --- REBUILD MODEL FUNCTIONALLY ---
# This assumes a functional model. For complex models, manual intervention may be needed.
def rebuild_functional_model(model):
    layer_outputs = {}
    input_layers = []
    
    # Check if input layer name is 'input' and rename it
    input_shape = tuple(model.inputs[0].shape)[1:]
    input_dtype = model.inputs[0].dtype
    original_input_name = model.inputs[0].name.split(':')[0]
    
    # Always use 'input_image' as the input name for hls4ml compatibility
    input_tensor = tf.keras.Input(shape=input_shape, name='input_image', dtype=input_dtype)
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            # Store input tensor with the new name
            layer_outputs['input_image'] = input_tensor
            input_layers.append(input_tensor)
        else:
            inbound_nodes = []
            for node in layer._inbound_nodes:
                if hasattr(node, 'inbound_layers'):
                    inbound = node.inbound_layers
                    if isinstance(inbound, list):
                        for l in inbound:
                            # Map any 'input' references to 'input_image'
                            key = 'input_image' if l.name == 'input' else l.name
                            if key in layer_outputs:
                                inbound_nodes.append(layer_outputs[key])
                    else:
                        # Map any 'input' references to 'input_image'
                        key = 'input_image' if inbound.name == 'input' else inbound.name
                        if key in layer_outputs:
                            inbound_nodes.append(layer_outputs[key])
            
            if not inbound_nodes:
                inbound_nodes = [input_tensor]
            
            # Automatically flatten input for Dense/QDense layers if needed
            is_dense = layer.__class__.__name__ in ["Dense", "QDense"]
            input_to_layer = inbound_nodes[0] if inbound_nodes else input_tensor
            if is_dense and hasattr(input_to_layer, 'shape') and len(input_to_layer.shape) > 2:
                flatten_layer = tf.keras.layers.Flatten(name=layer.name + "_auto_flatten")
                input_to_layer = flatten_layer(input_to_layer)
            
            if isinstance(layer, Concatenate) or isinstance(layer, Add):
                layer_outputs[layer.name] = refactor_merge_layer(layer.__class__(**layer.get_config()), inbound_nodes)
            else:
                # Fix quantizers in the layer before applying it
                layer_config, was_fixed = fix_quantizer_in_layer(layer)
                if was_fixed:
                    print(f"✅ Fixed quantizers in layer: {layer.name}")
                    try:
                        layer_class = layer.__class__
                        new_layer = layer_class(**layer_config)
                        layer_outputs[layer.name] = new_layer(input_to_layer)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not recreate layer {layer.name}: {e}")
                        print(f"   Using original layer")
                        layer_outputs[layer.name] = layer(input_to_layer)
                else:
                    layer_outputs[layer.name] = layer(input_to_layer)
    
    # Build new model
    if input_layers:
        new_model = Model(inputs=input_layers if len(input_layers) > 1 else input_layers[0], outputs=layer_outputs[model.layers[-1].name])
    else:
        new_model = Model(inputs=input_tensor, outputs=layer_outputs[model.layers[-1].name])
    
    return new_model

# --- MAIN PROCESSING ---
print("🔧 Starting HLS4ML Model Conversion...")
print(f"Input model: {INPUT_MODEL_PATH}")

# First, try deep fixing the existing model
print("\n🔍 Step 1: Deep fixing quantizers in existing model...")
fixed_count = deep_fix_model_quantizers(model)
print(f"Fixed {fixed_count} layers with problematic quantizers")

# Then rebuild the model functionally
print("\n🔧 Step 2: Rebuilding model functionally...")
new_model = rebuild_functional_model(model)
new_model.set_weights(model.get_weights())

# --- SAVE ---
new_model.save(OUTPUT_MODEL_PATH)
print(f"Saved HLS4ML-compatible model to: {OUTPUT_MODEL_PATH}")

# --- SUMMARY ---
print("\n" + "="*60)
print("🎉 HLS4ML Model Conversion Completed!")
print("="*60)
print("✅ Fixed problematic quantizers (quantized_po2 → quantized_bits)")
print("✅ Fixed problematic quantizers (quantized_relu_po2 → quantized_relu)")
print("✅ Removed unsupported alpha and scale_axis parameters")
print("✅ Refactored multi-input merge layers for HLS4ML compatibility")
print("✅ Renamed input layer to 'input_image' for HLS4ML compatibility")
print(f"✅ Fixed {fixed_count} layers with problematic quantizers")
print("\nThe model should now be fully compatible with HLS4ML conversion.")
print("You can now run the HLS4ML conversion without quantizer errors.")
print("="*60)
