#!/usr/bin/env python3
"""
Analyze TensorBoard histogram data to determine optimal quantization settings
"""

import os
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util import event_pb2
from tensorflow.core.framework import summary_pb2
import tensorflow as tf

def parse_histogram_proto(hist):
    """Parse histogram protobuf to get statistics"""
    if hist.bucket_limit:
        # Get bucket limits and counts
        bucket_limits = list(hist.bucket_limit)
        bucket_counts = list(hist.bucket)
        
        # Calculate statistics
        total_count = sum(bucket_counts)
        if total_count == 0:
            return None
            
        # Estimate min/max from buckets
        min_val = bucket_limits[0] if bucket_counts[0] > 0 else bucket_limits[1]
        max_val = bucket_limits[-1]
        
        # Find actual range with data
        first_nonzero = 0
        last_nonzero = len(bucket_counts) - 1
        
        for i, count in enumerate(bucket_counts):
            if count > 0:
                first_nonzero = i
                break
                
        for i in range(len(bucket_counts) - 1, -1, -1):
            if bucket_counts[i] > 0:
                last_nonzero = i
                break
        
        actual_min = bucket_limits[first_nonzero] if first_nonzero < len(bucket_limits) else min_val
        actual_max = bucket_limits[last_nonzero] if last_nonzero < len(bucket_limits) else max_val
        
        return {
            'min': float(actual_min),
            'max': float(actual_max),
            'bucket_limits': bucket_limits,
            'bucket_counts': bucket_counts,
            'total_count': total_count
        }
    return None

def recommend_quantization(min_val, max_val, layer_name=""):
    """Recommend quantization settings based on value range"""
    range_size = max_val - min_val
    abs_max = max(abs(min_val), abs(max_val))
    
    # Calculate required integer bits (add 1 for sign bit)
    if abs_max == 0:
        integer_bits = 1
    else:
        integer_bits = max(1, int(np.ceil(np.log2(abs_max + 1e-8))) + 1)
    
    # Quantization recommendations
    recommendations = {}
    
    # 8-bit recommendation
    if integer_bits <= 6:  # Leave at least 2 bits for fractional
        recommendations['8bit'] = {
            'bits': 8,
            'integer': integer_bits,
            'fractional': 8 - integer_bits,
            'range': [-2**(integer_bits-1), 2**(integer_bits-1) - 2**(-(8-integer_bits))],
            'precision': 2**(-(8-integer_bits))
        }
    
    # 16-bit recommendation
    if integer_bits <= 14:  # Leave at least 2 bits for fractional
        recommendations['16bit'] = {
            'bits': 16,
            'integer': integer_bits,
            'fractional': 16 - integer_bits,
            'range': [-2**(integer_bits-1), 2**(integer_bits-1) - 2**(-(16-integer_bits))],
            'precision': 2**(-(16-integer_bits))
        }
    
    # 32-bit recommendation (always possible)
    recommendations['32bit'] = {
        'bits': 32,
        'integer': integer_bits,
        'fractional': 32 - integer_bits,
        'range': [-2**(integer_bits-1), 2**(integer_bits-1) - 2**(-(32-integer_bits))],
        'precision': 2**(-(32-integer_bits))
    }
    
    return integer_bits, recommendations

def analyze_tensorboard_logs(log_dir):
    """Analyze all TensorBoard event files in the directory"""
    layer_stats = {}
    
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents') and file.endswith('.v2'):
                event_files.append(os.path.join(root, file))
    
    print(f"Found {len(event_files)} event files")
    
    # Process each event file
    for event_file in event_files:
        print(f"\nProcessing: {os.path.basename(event_file)}")
        
        try:
            for event in summary_iterator(event_file):
                if event.summary:
                    for value in event.summary.value:
                        if value.histo:
                            # This is a histogram
                            layer_name = value.tag
                            hist_data = parse_histogram_proto(value.histo)
                            
                            if hist_data:
                                if layer_name not in layer_stats:
                                    layer_stats[layer_name] = []
                                layer_stats[layer_name].append(hist_data)
                                
        except Exception as e:
            print(f"Error processing {event_file}: {e}")
            continue
    
    return layer_stats

def generate_quantization_config(layer_stats):
    """Generate optimized quantization configuration based on TensorBoard data"""
    print("\n" + "="*80)
    print("TENSORBOARD-BASED QUANTIZATION ANALYSIS")
    print("="*80)
    
    layer_recommendations = {}
    
    for layer_name, hist_list in layer_stats.items():
        if not hist_list:
            continue
            
        # Get latest histogram (most recent training data)
        latest_hist = hist_list[-1]
        min_val = latest_hist['min']
        max_val = latest_hist['max']
        
        print(f"\n=== Layer: {layer_name} ===")
        print(f"Value Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"Range Size: {max_val - min_val:.6f}")
        print(f"Histogram Samples: {len(hist_list)}")
        
        # Get quantization recommendations
        integer_bits, recommendations = recommend_quantization(min_val, max_val, layer_name)
        layer_recommendations[layer_name] = recommendations
        
        print(f"Required Integer Bits: {integer_bits}")
        
        for precision, config in recommendations.items():
            print(f"  {precision}: {config['bits']} bits ({config['integer']} int, {config['fractional']} frac)")
            print(f"    Range: [{config['range'][0]:.6f}, {config['range'][1]:.6f}]")
            print(f"    Precision: {config['precision']:.8f}")
    
    return layer_recommendations

def categorize_layers(layer_recommendations):
    """Categorize layers into precision groups based on requirements"""
    precision_groups = {
        '8bit': [],
        '16bit': [],
        '32bit': []
    }
    
    print("\n" + "="*80)
    print("LAYER PRECISION CATEGORIZATION")
    print("="*80)
    
    for layer_name, recommendations in layer_recommendations.items():
        # Determine minimum required precision
        if '8bit' in recommendations:
            precision_groups['8bit'].append((layer_name, recommendations['8bit']))
            print(f"{layer_name:40} -> 8-bit  (int: {recommendations['8bit']['integer']})")
        elif '16bit' in recommendations:
            precision_groups['16bit'].append((layer_name, recommendations['16bit']))
            print(f"{layer_name:40} -> 16-bit (int: {recommendations['16bit']['integer']})")
        else:
            precision_groups['32bit'].append((layer_name, recommendations['32bit']))
            print(f"{layer_name:40} -> 32-bit (int: {recommendations['32bit']['integer']})")
    
    return precision_groups

def generate_config_code(precision_groups):
    """Generate the quantization configuration code"""
    print("\n" + "="*80)
    print("OPTIMIZED QUANTIZATION CONFIGURATION")
    print("="*80)
    
    # Find the most common integer bit requirements for each precision
    int_bits_8 = [config['integer'] for _, config in precision_groups['8bit']]
    int_bits_16 = [config['integer'] for _, config in precision_groups['16bit']]
    int_bits_32 = [config['integer'] for _, config in precision_groups['32bit']]
    
    # Use the maximum required integer bits for each precision to be safe
    optimal_8_int = max(int_bits_8) if int_bits_8 else 2
    optimal_16_int = max(int_bits_16) if int_bits_16 else 3
    optimal_32_int = max(int_bits_32) if int_bits_32 else 4
    
    print(f"\nOptimal Integer Bits:")
    print(f"  8-bit layers:  {optimal_8_int} integer bits")
    print(f"  16-bit layers: {optimal_16_int} integer bits")
    print(f"  32-bit layers: {optimal_32_int} integer bits")
    
    config_code = f'''
# Optimized quantization config based on TensorBoard histogram analysis
quantized_bits_8_config = {{
    'bits': 8,
    'integer': {optimal_8_int},  # Optimized based on actual weight distributions
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_8_config = {{
    'bits': 8,
    'integer': {optimal_8_int},
    'use_stochastic_rounding': True
}}

quantized_bits_16_config = {{
    'bits': 16,
    'integer': {optimal_16_int},  # Optimized based on actual weight distributions
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_16_config = {{
    'bits': 16,
    'integer': {optimal_16_int},
    'use_stochastic_rounding': True
}}

quantized_bits_32_config = {{
    'bits': 32,
    'integer': {optimal_32_int},  # Optimized based on actual weight distributions
    'symmetric': True,
    'alpha': 'auto_po2',
    'use_stochastic_rounding': True,
    'scale_axis': None
}}

quantized_relu_32_config = {{
    'bits': 32,
    'integer': {optimal_32_int},
    'use_stochastic_rounding': True
}}'''
    
    print(config_code)
    return optimal_8_int, optimal_16_int, optimal_32_int

if __name__ == "__main__":
    # Analyze TensorBoard logs
    log_dir = "models/tb_logs"
    print(f"Analyzing TensorBoard logs in: {log_dir}")
    
    layer_stats = analyze_tensorboard_logs(log_dir)
    
    if not layer_stats:
        print("No histogram data found in TensorBoard logs!")
        print("Make sure histogram logging is enabled during training.")
        exit(1)
    
    print(f"\nFound histogram data for {len(layer_stats)} layers")
    
    # Generate recommendations
    layer_recommendations = generate_quantization_config(layer_stats)
    
    # Categorize layers
    precision_groups = categorize_layers(layer_recommendations)
    
    # Generate configuration code
    optimal_8_int, optimal_16_int, optimal_32_int = generate_config_code(precision_groups)
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"  8-bit layers ({len(precision_groups['8bit'])}): integer bits = {optimal_8_int}")
    print(f"  16-bit layers ({len(precision_groups['16bit'])}): integer bits = {optimal_16_int}")
    print(f"  32-bit layers ({len(precision_groups['32bit'])}): integer bits = {optimal_32_int}")
    print(f"{'='*80}")