#!/usr/bin/env python3
"""
Weight Analysis Script for HLS4ML Quantization
==============================================

This script analyzes the weights of a trained model and generates a text report
showing the min/max ranges for each layer. This information helps in setting
appropriate quantized_bits_ parameters for each layer.

Usage:
    python analyze_weights.py --model_path ./models/your_model.keras
"""

import os
import numpy as np
import tensorflow as tf
import argparse
import glob
import time
from pathlib import Path

def analyze_model_weights(model_path):
    """
    Analyze weights of a trained model and generate a text report.
    
    Args:
        model_path: Path to the trained model file
    """
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully: {model.name}")
        print(f"Total layers: {len(model.layers)}")
        print("=" * 80)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Analyze each layer with weights
    weight_layers = []
    
    for layer in model.layers:
        if hasattr(layer, 'weights') and layer.weights:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'weights': []
            }
            
            for i, weight in enumerate(layer.weights):
                weight_np = weight.numpy()
                weight_info = {
                    'name': weight.name,
                    'shape': weight_np.shape,
                    'min': float(np.min(weight_np)),
                    'max': float(np.max(weight_np)),
                    'mean': float(np.mean(weight_np)),
                    'std': float(np.std(weight_np)),
                    'abs_max': float(np.max(np.abs(weight_np)))
                }
                layer_info['weights'].append(weight_info)
            
            weight_layers.append(layer_info)
    
    # Generate report
    print("WEIGHT ANALYSIS REPORT")
    print("=" * 80)
    print(f"Model: {model.name}")
    print(f"Total layers with weights: {len(weight_layers)}")
    print("=" * 80)
    
    for layer_info in weight_layers:
        print(f"\nLayer: {layer_info['name']}")
        print(f"Type: {layer_info['type']}")
        print("-" * 60)
        
        for weight_info in layer_info['weights']:
            print(f"  Weight: {weight_info['name']}")
            print(f"    Shape: {weight_info['shape']}")
            print(f"    Range: [{weight_info['min']:.6f}, {weight_info['max']:.6f}]")
            print(f"    Mean: {weight_info['mean']:.6f}")
            print(f"    Std: {weight_info['std']:.6f}")
            print(f"    Abs Max: {weight_info['abs_max']:.6f}")
            
            # Suggest quantization bits based on range
            abs_max = weight_info['abs_max']
            if abs_max <= 1.0:
                suggested_bits = 8
            elif abs_max <= 4.0:
                suggested_bits = 10
            elif abs_max <= 16.0:
                suggested_bits = 12
            else:
                suggested_bits = 16
            
            print(f"    Suggested bits: {suggested_bits}")
            print()
    
    # Summary for QKeras configuration
    print("=" * 80)
    print("QKERAS CONFIGURATION SUGGESTIONS")
    print("=" * 80)
    
    for layer_info in weight_layers:
        layer_name = layer_info['name']
        max_abs_value = max([w['abs_max'] for w in layer_info['weights']])
        
        # Determine suggested configuration
        if max_abs_value <= 1.0:
            bits = 8
            int_bits = 1
        elif max_abs_value <= 2.0:
            bits = 8
            int_bits = 2
        elif max_abs_value <= 4.0:
            bits = 10
            int_bits = 3
        elif max_abs_value <= 8.0:
            bits = 12
            int_bits = 4
        else:
            bits = 16
            int_bits = 5
        
        print(f"{layer_name}: quantized_bits(bits={bits}, integer={int_bits}, keep_negative=True)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")

def find_latest_model(models_dir="./models"):
    """
    Find the latest model checkpoint in the models directory.
    
    Args:
        models_dir: Directory containing model files
    
    Returns:
        Path to the latest model file or None if not found
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return None
    
    # Look for common model file patterns
    patterns = [
        "*.keras",
        "*.h5",
        "*checkpoint*.keras",
        "*checkpoint*.h5",
        "*model*.keras",
        "*model*.h5"
    ]
    
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(os.path.join(models_dir, pattern)))
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return None
    
    # Sort by modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files[:5]):  # Show top 5
        mtime = os.path.getmtime(model_file)
        print(f"  {i+1}. {os.path.basename(model_file)} (modified: {time.ctime(mtime)})")
    
    return model_files[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model weights for quantization')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to the model file to analyze')
    parser.add_argument('--models_dir', type=str, default='./models',
                       help='Directory to search for models if model_path not specified')
    
    args = parser.parse_args()
    
    # Determine which model to analyze
    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            exit(1)
    else:
        model_path = find_latest_model(args.models_dir)
        if not model_path:
            print("No model found to analyze. Please specify --model_path or ensure models exist in --models_dir")
            exit(1)
    
    # Analyze the model
    analyze_model_weights(model_path)
