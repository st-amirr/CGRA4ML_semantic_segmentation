#!/usr/bin/env python3
"""
Extract weight statistics directly from a trained model to optimize quantization
"""

import os
import numpy as np
import tensorflow as tf

def analyze_layer_weights(model):
    """Analyze weights from a trained model"""
    layer_stats = {}
    
    print("="*80)
    print("MODEL WEIGHT ANALYSIS FOR QUANTIZATION OPTIMIZATION")
    print("="*80)
    
    for layer in model.layers:
        if hasattr(layer, 'weights') and layer.weights:
            for weight in layer.weights:
                weight_name = f"{layer.name}/{weight.name}"
                weight_values = weight.numpy()
                
                if weight_values.size == 0:
                    continue
                    
                # Calculate statistics
                min_val = float(np.min(weight_values))
                max_val = float(np.max(weight_values))
                mean_val = float(np.mean(weight_values))
                std_val = float(np.std(weight_values))
                abs_max = max(abs(min_val), abs(max_val))
                
                # Calculate required integer bits (add 1 for sign bit)
                if abs_max == 0:
                    integer_bits = 1
                else:
                    integer_bits = max(1, int(np.ceil(np.log2(abs_max + 1e-8))) + 1)
                
                layer_stats[weight_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'abs_max': abs_max,
                    'integer_bits': integer_bits,
                    'shape': weight_values.shape,
                    'size': weight_values.size
                }
                
                print(f"\n=== {weight_name} ===")
                print(f"Shape: {weight_values.shape}")
                print(f"Range: [{min_val:.6f}, {max_val:.6f}]")
                print(f"Abs Max: {abs_max:.6f}")
                print(f"Required Integer Bits: {integer_bits}")
                print(f"Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    
    return layer_stats

def categorize_layers_by_name(layer_stats):
    """Categorize layers based on naming patterns and weight requirements"""
    categories = {
        'input_light': [],      # Input and light encoder layers (8-bit)
        'encoder_medium': [],   # Medium encoder layers (16-bit)
        'bridge_critical': [],  # Bridge and critical decoder layers (32-bit)
        'decoder_medium': [],   # Medium decoder layers (16-bit)
        'output_light': []      # Output layers (8-bit)
    }
    
    for weight_name, stats in layer_stats.items():
        layer_name = weight_name.split('/')[0]
        
        # Categorize based on layer naming patterns from trainQAT.py
        if ('init_conv' in layer_name or 
            'enc1_' in layer_name or 
            'final_conv' in layer_name or 
            'output' in layer_name):
            categories['input_light'].append((weight_name, stats))
            
        elif ('enc2_' in layer_name or 
              'enc3_' in layer_name or
              'dec1_' in layer_name or 
              'dec2_' in layer_name):
            categories['encoder_medium'].append((weight_name, stats))
            
        elif ('bridge_' in layer_name or 
              'dec3_' in layer_name):
            categories['bridge_critical'].append((weight_name, stats))
            
        else:
            # Default categorization based on integer bit requirements
            if stats['integer_bits'] <= 3:
                categories['input_light'].append((weight_name, stats))
            elif stats['integer_bits'] <= 5:
                categories['encoder_medium'].append((weight_name, stats))
            else:
                categories['bridge_critical'].append((weight_name, stats))
    
    return categories

def calculate_optimal_quantization(categories):
    """Calculate optimal quantization settings for each category"""
    print("\n" + "="*80)
    print("OPTIMAL QUANTIZATION ANALYSIS")
    print("="*80)
    
    optimal_configs = {}
    
    for category_name, layers in categories.items():
        if not layers:
            continue
            
        print(f"\n--- {category_name.upper()} ---")
        
        # Find maximum integer bits required in this category
        max_int_bits = max(stats['integer_bits'] for _, stats in layers)
        
        # Determine optimal bit width
        if category_name == 'input_light' or category_name == 'output_light':
            target_bits = 8
        elif category_name == 'encoder_medium' or category_name == 'decoder_medium':
            target_bits = 16
        else:  # bridge_critical
            target_bits = 32
        
        # Ensure we have enough integer bits
        optimal_int_bits = min(max_int_bits, target_bits - 2)  # Leave at least 2 fractional bits
        fractional_bits = target_bits - optimal_int_bits
        
        optimal_configs[category_name] = {
            'bits': target_bits,
            'integer': optimal_int_bits,
            'fractional': fractional_bits,
            'layer_count': len(layers)
        }
        
        print(f"Layers: {len(layers)}")
        print(f"Max required integer bits: {max_int_bits}")
        print(f"Optimal config: {target_bits}-bit ({optimal_int_bits} int, {fractional_bits} frac)")
        
        # Show some example layers
        for i, (weight_name, stats) in enumerate(layers[:3]):
            print(f"  Example: {weight_name} - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        if len(layers) > 3:
            print(f"  ... and {len(layers) - 3} more layers")
    
    return optimal_configs

def generate_quantization_code(optimal_configs):
    """Generate the optimized quantization configuration code"""
    print("\n" + "="*80)
    print("GENERATED QUANTIZATION CONFIGURATION")
    print("="*80)
    
    # Map category to config variable names
    config_mapping = {
        'input_light': ('8', optimal_configs.get('input_light', {}).get('integer', 2)),
        'encoder_medium': ('16', optimal_configs.get('encoder_medium', {}).get('integer', 3)),
        'bridge_critical': ('32', optimal_configs.get('bridge_critical', {}).get('integer', 4))
    }
    
    # Use the most conservative (highest) integer bits from light categories
    light_int_bits = max(
        optimal_configs.get('input_light', {}).get('integer', 2),
        optimal_configs.get('output_light', {}).get('integer', 2)
    )
    
    medium_int_bits = max(
        optimal_configs.get('encoder_medium', {}).get('integer', 3),
        optimal_configs.get('decoder_medium', {}).get('integer', 3)
    )
    
    critical_int_bits = optimal_configs.get('bridge_critical', {}).get('integer', 4)
    
    config_code = f'''
# Optimized quantization config based on actual trained model weights
quantized_bits_8_config = {{
    'bits': 8,
    'integer': {light_int_bits},  # Based on analysis of input/output layers
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_8_config = {{
    'bits': 8,
    'integer': {light_int_bits},
    'use_stochastic_rounding': True
}}

quantized_bits_16_config = {{
    'bits': 16,
    'integer': {medium_int_bits},  # Based on analysis of encoder/decoder layers
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_16_config = {{
    'bits': 16,
    'integer': {medium_int_bits},
    'use_stochastic_rounding': True
}}

quantized_bits_32_config = {{
    'bits': 32,
    'integer': {critical_int_bits},  # Based on analysis of bridge/critical layers
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_32_config = {{
    'bits': 32,
    'integer': {critical_int_bits},
    'use_stochastic_rounding': True
}}'''
    
    print(config_code)
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"  8-bit config:  {light_int_bits} integer bits ({8-light_int_bits} fractional)")
    print(f"  16-bit config: {medium_int_bits} integer bits ({16-medium_int_bits} fractional)")
    print(f"  32-bit config: {critical_int_bits} integer bits ({32-critical_int_bits} fractional)")
    print(f"{'='*80}")
    
    return light_int_bits, medium_int_bits, critical_int_bits

def main():
    # Analyze specific model
    model_dir = "/app/models"
    target_model = "float32_model_checkpoint_epoch_70.keras"
    model_path = os.path.join(model_dir, target_model)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Analyzing model: {target_model}")
    
    # Import the model creation function to recreate architecture
    import sys
    sys.path.append('/workspace')
    from trainQAT import create_segmentation_model
    
    # Load model by recreating architecture and loading weights
    try:
        # Create the model architecture (float32 version)
        model = create_segmentation_model(use_quantization=False)
        print(f"Model architecture created: {len(model.layers)} layers")
        
        # Load the weights from checkpoint
        model.load_weights(model_path)
        print("Weights loaded successfully from checkpoint")
        
    except Exception as e:
        print(f"Error loading model/weights: {e}")
        print("Trying direct model loading...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded directly: {len(model.layers)} layers")
        except Exception as e2:
            print(f"Both methods failed: {e2}")
            return
    
    # Analyze weights
    layer_stats = analyze_layer_weights(model)
    
    if not layer_stats:
        print("No weights found in model!")
        return
    
    # Categorize layers
    categories = categorize_layers_by_name(layer_stats)
    
    # Calculate optimal quantization
    optimal_configs = calculate_optimal_quantization(categories)
    
    # Generate configuration code
    generate_quantization_code(optimal_configs)

if __name__ == "__main__":
    main()