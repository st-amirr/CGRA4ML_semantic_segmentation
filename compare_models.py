#!/usr/bin/env python3
"""
Model Comparison and Analysis Script
===================================

This script helps compare float32 and QAT models, analyze their performance,
and prepare for hls4ml export.

Usage:
    python compare_models.py --float32_model=/app/models/float32_best.keras --qat_model=/app/models/qat_best.keras
"""

import os
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from qkeras.utils import model_save_quantized_weights

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare Float32 and QAT models')
    parser.add_argument('--float32_model', type=str, required=True,
                       help='Path to float32 model')
    parser.add_argument('--qat_model', type=str, required=True,
                       help='Path to QAT model')
    parser.add_argument('--test_image', type=str, default='',
                       help='Path to test image for inference comparison')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                       help='Directory to save analysis results')
    return parser.parse_args()

def load_model_safely(model_path):
    """Load model with error handling"""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Successfully loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {e}")
        return None

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return total_params, trainable_params

def analyze_model_architecture(model, model_name):
    """Analyze model architecture and layers"""
    print(f"\n📊 {model_name} Architecture Analysis")
    print("=" * 50)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Count layer types
    layer_types = {}
    quantized_layers = 0
    
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        if 'Q' in layer_type:  # QKeras layers start with Q
            quantized_layers += 1
    
    print(f"Total layers: {len(model.layers)}")
    print(f"Quantized layers: {quantized_layers}")
    print(f"Quantization ratio: {quantized_layers/len(model.layers)*100:.1f}%")
    
    print("\nLayer type distribution:")
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_layers': len(model.layers),
        'quantized_layers': quantized_layers,
        'layer_types': layer_types
    }

def estimate_model_size(model):
    """Estimate model size in MB"""
    total_params, _ = count_parameters(model)
    
    # Check if model has quantized layers
    has_quantized = any('Q' in type(layer).__name__ for layer in model.layers)
    
    if has_quantized:
        # Estimate based on quantization (average 8-16 bits)
        avg_bits = 12  # Average between 8 and 16 bit quantization
        size_mb = (total_params * avg_bits) / (8 * 1024 * 1024)
    else:
        # Float32 model
        size_mb = (total_params * 32) / (8 * 1024 * 1024)
    
    return size_mb

def compare_inference_speed(float32_model, qat_model, input_shape=(1, 512, 1024, 3)):
    """Compare inference speed between models"""
    print(f"\n⚡ Inference Speed Comparison")
    print("=" * 40)
    
    # Create dummy input
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # Warm up models
    print("Warming up models...")
    for _ in range(3):
        _ = float32_model(dummy_input, training=False)
        _ = qat_model(dummy_input, training=False)
    
    # Time float32 model
    import time
    num_runs = 10
    
    start_time = time.time()
    for _ in range(num_runs):
        _ = float32_model(dummy_input, training=False)
    float32_time = (time.time() - start_time) / num_runs
    
    # Time QAT model
    start_time = time.time()
    for _ in range(num_runs):
        _ = qat_model(dummy_input, training=False)
    qat_time = (time.time() - start_time) / num_runs
    
    print(f"Float32 inference time: {float32_time*1000:.2f} ms")
    print(f"QAT inference time: {qat_time*1000:.2f} ms")
    print(f"Speed ratio (QAT/Float32): {qat_time/float32_time:.2f}x")
    
    return float32_time, qat_time

def compare_outputs(float32_model, qat_model, test_image_path=None, output_dir='./analysis'):
    """Compare model outputs on test image"""
    print(f"\n🖼️  Output Comparison")
    print("=" * 30)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if test_image_path and os.path.exists(test_image_path):
        # Load and preprocess test image
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 512))
        image_norm = image.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_norm, axis=0)
        
        # Get predictions
        float32_pred = float32_model(image_batch, training=False)
        qat_pred = qat_model(image_batch, training=False)
        
        # Convert to class predictions
        float32_classes = np.argmax(float32_pred[0], axis=-1)
        qat_classes = np.argmax(qat_pred[0], axis=-1)
        
        # Calculate pixel-wise agreement
        agreement = np.mean(float32_classes == qat_classes)
        print(f"Pixel-wise agreement: {agreement*100:.2f}%")
        
        # Calculate prediction confidence differences
        float32_conf = np.max(float32_pred[0], axis=-1)
        qat_conf = np.max(qat_pred[0], axis=-1)
        conf_diff = np.mean(np.abs(float32_conf - qat_conf))
        print(f"Average confidence difference: {conf_diff:.4f}")
        
        # Save visualizations
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(float32_classes, cmap='tab20')
        plt.title('Float32 Prediction')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(qat_classes, cmap='tab20')
        plt.title('QAT Prediction')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(float32_conf, cmap='viridis')
        plt.title('Float32 Confidence')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(qat_conf, cmap='viridis')
        plt.title('QAT Confidence')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        difference = np.abs(float32_classes.astype(int) - qat_classes.astype(int))
        plt.imshow(difference, cmap='Reds')
        plt.title('Prediction Differences')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to: {output_dir}/model_comparison.png")
        
        return agreement, conf_diff
    else:
        # Create dummy input for basic comparison
        dummy_input = np.random.random((1, 512, 1024, 3)).astype(np.float32)
        
        float32_pred = float32_model(dummy_input, training=False)
        qat_pred = qat_model(dummy_input, training=False)
        
        # Calculate output correlation
        correlation = np.corrcoef(float32_pred.numpy().flatten(), qat_pred.numpy().flatten())[0, 1]
        print(f"Output correlation: {correlation:.4f}")
        
        return correlation, 0

def export_for_hls4ml(qat_model, output_dir='./analysis'):
    """Export QAT model for hls4ml"""
    print(f"\n🔧 Exporting for hls4ml")
    print("=" * 30)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save quantized weights
        quantized_weights_path = os.path.join(output_dir, 'quantized_weights_for_hls4ml.h5')
        model_save_quantized_weights(qat_model, quantized_weights_path)
        print(f"✅ Quantized weights saved: {quantized_weights_path}")
        
        # Save model architecture as JSON
        model_json = qat_model.to_json()
        json_path = os.path.join(output_dir, 'model_architecture.json')
        with open(json_path, 'w') as f:
            f.write(model_json)
        print(f"✅ Model architecture saved: {json_path}")
        
        # Create hls4ml config template
        hls4ml_config = """
# HLS4ML Configuration Template
# Copy and modify this for your hls4ml synthesis

import hls4ml
from tensorflow.keras.models import model_from_json

# Load your QAT model
with open('model_architecture.json', 'r') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights('quantized_weights_for_hls4ml.h5')

# Configure hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Set your target FPGA (adjust as needed)
config['Model']['Strategy'] = 'Resource'  # or 'Latency'
config['Model']['Precision'] = 'ap_fixed<16,6>'  # Adjust based on your quantization

# Configure specific layers if needed
# config['LayerName']['Strategy'] = 'Resource'
# config['LayerName']['ReuseFactor'] = 1

# Convert to HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    model, 
    hls_config=config,
    output_dir='hls4ml_output',
    backend='Vivado'  # or 'VivadoAccelerator' for board deployment
)

# Compile (this will take time)
hls_model.compile()

# Test inference
# y_hls = hls_model.predict(X_test)
"""
        
        config_path = os.path.join(output_dir, 'hls4ml_conversion_template.py')
        with open(config_path, 'w') as f:
            f.write(hls4ml_config)
        print(f"✅ HLS4ML template saved: {config_path}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def generate_report(float32_stats, qat_stats, speed_comparison, output_comparison, output_dir):
    """Generate comprehensive comparison report"""
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("HLS4ML Model Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("📊 Model Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Float32 Model:\n")
        f.write(f"  - Total parameters: {float32_stats['total_params']:,}\n")
        f.write(f"  - Trainable parameters: {float32_stats['trainable_params']:,}\n")
        f.write(f"  - Total layers: {float32_stats['total_layers']}\n")
        f.write(f"  - Estimated size: {estimate_model_size(None):.2f} MB\n\n")
        
        f.write(f"QAT Model:\n")
        f.write(f"  - Total parameters: {qat_stats['total_params']:,}\n")
        f.write(f"  - Trainable parameters: {qat_stats['trainable_params']:,}\n")
        f.write(f"  - Total layers: {qat_stats['total_layers']}\n")
        f.write(f"  - Quantized layers: {qat_stats['quantized_layers']}\n")
        f.write(f"  - Quantization ratio: {qat_stats['quantized_layers']/qat_stats['total_layers']*100:.1f}%\n\n")
        
        if speed_comparison:
            f.write("⚡ Performance Comparison\n")
            f.write("-" * 25 + "\n")
            f.write(f"Float32 inference time: {speed_comparison[0]*1000:.2f} ms\n")
            f.write(f"QAT inference time: {speed_comparison[1]*1000:.2f} ms\n")
            f.write(f"Speed ratio: {speed_comparison[1]/speed_comparison[0]:.2f}x\n\n")
        
        if output_comparison:
            f.write("🎯 Output Quality\n")
            f.write("-" * 15 + "\n")
            if len(output_comparison) == 2:
                f.write(f"Pixel-wise agreement: {output_comparison[0]*100:.2f}%\n")
                f.write(f"Average confidence difference: {output_comparison[1]:.4f}\n\n")
            else:
                f.write(f"Output correlation: {output_comparison[0]:.4f}\n\n")
        
        f.write("🚀 Recommendations\n")
        f.write("-" * 15 + "\n")
        f.write("1. If Float32 model performance is satisfactory, proceed with QAT\n")
        f.write("2. Use the quantized weights for hls4ml synthesis\n")
        f.write("3. Test FPGA implementation with representative data\n")
        f.write("4. Monitor resource utilization during synthesis\n")
        f.write("5. Consider adjusting quantization parameters if needed\n")
    
    print(f"📋 Comprehensive report saved: {report_path}")

def main():
    args = parse_arguments()
    
    print("🔍 HLS4ML Model Comparison and Analysis")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    float32_model = load_model_safely(args.float32_model)
    qat_model = load_model_safely(args.qat_model)
    
    if not float32_model or not qat_model:
        print("❌ Failed to load one or both models. Exiting.")
        return
    
    # Analyze architectures
    float32_stats = analyze_model_architecture(float32_model, "Float32")
    qat_stats = analyze_model_architecture(qat_model, "QAT")
    
    # Compare inference speed
    speed_comparison = compare_inference_speed(float32_model, qat_model)
    
    # Compare outputs
    output_comparison = compare_outputs(
        float32_model, qat_model, 
        args.test_image, args.output_dir
    )
    
    # Export for hls4ml
    export_for_hls4ml(qat_model, args.output_dir)
    
    # Generate comprehensive report
    generate_report(
        float32_stats, qat_stats, 
        speed_comparison, output_comparison, 
        args.output_dir
    )
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
