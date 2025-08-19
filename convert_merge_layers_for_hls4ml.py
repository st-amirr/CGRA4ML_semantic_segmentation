import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Add
import h5py

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
                layer_outputs[layer.name] = layer(input_to_layer)
    
    # Build new model
    if input_layers:
        new_model = Model(inputs=input_layers if len(input_layers) > 1 else input_layers[0], outputs=layer_outputs[model.layers[-1].name])
    else:
        new_model = Model(inputs=input_tensor, outputs=layer_outputs[model.layers[-1].name])
    
    return new_model

new_model = rebuild_functional_model(model)
new_model.set_weights(model.get_weights())

# --- SAVE ---
new_model.save(OUTPUT_MODEL_PATH)
print(f"Saved HLS4ML-compatible model to: {OUTPUT_MODEL_PATH}")
