#!/usr/bin/env python3
"""
HLS4ML FPGA Model Builder for LMIINet Semantic Segmentation
===========================================================

This script converts a trained LMIINet model to HLS4ML format and generates
FPGA implementation code. It handles the complete pipeline from model loading
to HLS4ML code generation.

Usage:
    python build_fpga_model.py --model_path ./models/qat_model_final.keras
    python build_fpga_model.py --model_path ./models/float32_model_final.keras --quantized

Features:
- Loads trained LMIINet models (float32 or QAT)
- Converts to HLS4ML-compatible format
- Generates FPGA implementation with Vivado HLS
- Configurable precision and resource constraints
- Comprehensive validation and testing
- Performance analysis and resource estimation

Requirements:
- hls4ml
- tensorflow
- qkeras (for QAT models)
- vivado_hls (for FPGA synthesis)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import hls4ml
from pathlib import Path
import json
import time
from datetime import datetime

# Import model architecture from training script
from trainQAT import (
    create_segmentation_model,
    create_hls4ml_export_model,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FPGA model using HLS4ML")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--quantized", action="store_true", help="Model is quantized (QAT model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fpga_models",
        help="Output directory for FPGA model",
    )
    parser.add_argument(
        "--part", type=str, default="xczu7ev-ffvc1156-2-e", help="FPGA part number"
    )
    parser.add_argument(
        "--clock_period", type=float, default=5.0, help="Clock period in nanoseconds"
    )
    parser.add_argument(
        "--io_type",
        type=str,
        default="io_parallel",
        choices=["io_parallel", "io_stream"],
        help="I/O type for HLS4ML",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="Latency",
        choices=["Latency", "Resource"],
        help="HLS4ML optimization strategy",
    )
    parser.add_argument(
        "--reuse_factor",
        type=int,
        default=1,
        help="Reuse factor for resource optimization",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="ap_fixed<16,6>",
        help="Precision for fixed-point arithmetic (used for float32 models or as fallback)",
    )
    parser.add_argument(
        "--override_precision",
        type=str,
        default=None,
        help="Override all precisions with single value (for testing)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model conversion with test data",
    )
    parser.add_argument(
        "--synthesize", action="store_true", help="Run Vivado HLS synthesis"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def load_trained_model(model_path, is_quantized=False):
    """
    Load trained model and prepare for HLS4ML conversion

    Args:
        model_path: Path to the trained model
        is_quantized: Whether the model is quantized (QAT)

    Returns:
        Loaded and prepared model
    """
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Enforce .h5 format for QKeras models
        custom_objects = None
        if is_quantized:
            if model_path.endswith('.keras'):
                raise ValueError("QKeras models must be saved and loaded in .h5 format. Please re-save your model as .h5 and use that file.")
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
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print(f"Model loaded successfully: {model.name}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        # Create clean model for HLS4ML export (remove dropout layers)
        print("Creating clean model for HLS4ML export...")
        clean_model = create_hls4ml_export_model(
            trained_model=model, use_quantization=is_quantized
        )

        # Verify model compatibility
        verify_model_compatibility(clean_model)

        return clean_model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def verify_model_compatibility(model):
    """
    Verify that the model is compatible with HLS4ML

    Args:
        model: Keras model to verify
    """
    print("Verifying HLS4ML compatibility...")

    # Check for unsupported layers
    unsupported_layers = []
    supported_layers = [
        "InputLayer",
        "Conv2D",
        "DepthwiseConv2D",
        "BatchNormalization",
        "Activation",
        "Dense",
        "GlobalAveragePooling2D",
        "Reshape",
        "Add",
        "Concatenate",
        "UpSampling2D",
        "Dropout",
    ]

    for layer in model.layers:
        layer_type = type(layer).__name__
        if layer_type not in supported_layers:
            unsupported_layers.append(f"{layer.name} ({layer_type})")

    if unsupported_layers:
        print("⚠️  Warning: Found potentially unsupported layers:")
        for layer in unsupported_layers:
            print(f"   - {layer}")
        print("   These layers may cause issues during HLS4ML conversion.")
    else:
        print("✅ All layers are HLS4ML-compatible")

    # Check input/output shapes
    input_shape = (
        model.input_shape[0][1:]
        if isinstance(model.input_shape, list)
        else model.input_shape[1:]
    )
    output_shape = (
        model.output_shape[1:]
        if isinstance(model.output_shape, tuple)
        else model.output_shape[0][1:]
    )

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

    if input_shape != (INPUT_HEIGHT, INPUT_WIDTH, 3):
        print(
            f"⚠️  Warning: Expected input shape {(INPUT_HEIGHT, INPUT_WIDTH, 3)}, got {input_shape}"
        )

    if output_shape != (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES):
        print(
            f"⚠️  Warning: Expected output shape {(INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)}, got {output_shape}"
        )


def create_hls4ml_config(args):
    """
    Create HLS4ML configuration with multiple precision support

    Args:
        args: Command line arguments

    Returns:
        HLS4ML configuration dictionary
    """
    # Get input layer name from model if available
    input_layer_name = None
    if hasattr(args, 'model') and args.model is not None:
        input_layer_name = args.model.inputs[0].name.split(':')[0]
    else:
        input_layer_name = 'input_image'  # fallback
    config = {
        "Backend": "Vivado",
        "Model": {
            "Precision": args.precision,
            "ReuseFactor": args.reuse_factor,
            "Strategy": args.strategy,
        },
        "LayerName": {
            input_layer_name: {
                "IOType": args.io_type
            }
        },
        "LayerType": {},
        "Optimizers": {
            "eliminate_linear_activation": True,
            "fuse_batch_norm_quantized_tanh": True,
            "fuse_consecutive_batch_norm": True,
        },
        "SkipOptimizers": [],
        "LayerConfig": {},
    }

    # Define precision configurations based on model type
    if args.quantized:
        # QAT model - use the optimized precisions from your weight analysis
        conv_precision = "ap_fixed<32,8>"  # Conv layers: 32-bit (8 int, 24 frac) for maximum precision
        bn_precision = (
            "ap_fixed<16,4>"  # BatchNorm: 16-bit (4 int, 12 frac) for stability
        )
        dense_precision = (
            "ap_fixed<16,4>"  # Dense/MLP: 16-bit (4 int, 12 frac) for efficiency
        )
        output_precision = (
            "ap_fixed<32,8>"  # Output: 32-bit (8 int, 24 frac) for critical predictions
        )
        activation_precision = (
            "ap_fixed<32,16>"  # Activations: 32-bit (16 int, 16 frac) for accuracy
        )
    else:
        # Float32 model - use consistent precision
        conv_precision = args.precision
        bn_precision = args.precision
        dense_precision = args.precision
        output_precision = args.precision
        activation_precision = args.precision

    # Apply override if specified (for testing purposes)
    if args.override_precision:
        conv_precision = args.override_precision
        bn_precision = args.override_precision
        dense_precision = args.override_precision
        output_precision = args.override_precision
        activation_precision = args.override_precision

    # Add layer-specific configurations with multiple precisions
    config["LayerType"]["Conv2D"] = {
        "Precision": conv_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["QConv2D"] = {
        "Precision": conv_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["DepthwiseConv2D"] = {
        "Precision": conv_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["QDepthwiseConv2D"] = {
        "Precision": conv_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["Dense"] = {
        "Precision": dense_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["QDense"] = {
        "Precision": dense_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    config["LayerType"]["BatchNormalization"] = {
        "Precision": bn_precision,
    }

    config["LayerType"]["QBatchNormalization"] = {
        "Precision": bn_precision,
    }

    config["LayerType"]["Activation"] = {
        "Precision": activation_precision,
    }

    config["LayerType"]["QActivation"] = {
        "Precision": activation_precision,
    }

    # Layer-specific configurations for critical layers
    config["LayerName"]["output_conv"] = {
        "Precision": output_precision,
        "ReuseFactor": args.reuse_factor,
        "Strategy": args.strategy,
    }

    # Print precision configuration
    if args.quantized:
        print("Using QAT-optimized multiple precision configuration:")
        print(f"  Conv2D/DepthwiseConv2D: {conv_precision}")
        print(f"  BatchNormalization: {bn_precision}")
        print(f"  Dense/MLP: {dense_precision}")
        print(f"  Output layer: {output_precision}")
        print(f"  Activations: {activation_precision}")
    else:
        print(f"Using uniform precision: {args.precision}")

    return config


def convert_to_hls4ml(model, config, output_dir):

    """
    Convert Keras model to HLS4ML format

    Args:
        model: Keras model
        config: HLS4ML configuration
        output_dir: Output directory

    Returns:
        HLS4ML model
    """
    print("Converting model to HLS4ML format...")
    print("HLS4ML config:")
    import pprint
    pprint.pprint(config)
    print("Model input layers:")
    for layer in model.layers:
        if hasattr(layer, 'input_shape'):
            print(f"Layer name: {layer.name}, type: {layer.__class__.__name__}, input_shape: {layer.input_shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set part, clock_period, io_type in config
    config['FPGA'] = {
        'Part': args.part,
        'ClockPeriod': args.clock_period,
        'IOType': args.io_type
    }
    # For hls4ml 1.1.0, add model to config
    config['KerasModel'] = model
    # Add HLSConfig key for hls4ml 1.1.0 compatibility
    config['HLSConfig'] = {
        'Model': config.get('Model', {}),
        'LayerName': config.get('LayerName', {}),
        'LayerType': config.get('LayerType', {}),
        'Optimizers': config.get('Optimizers', {}),
        'SkipOptimizers': config.get('SkipOptimizers', []),
        'LayerConfig': config.get('LayerConfig', {}),
        'FPGA': config.get('FPGA', {}),
    }
    # Remove SkipOptimizers if Optimizers is present (hls4ml expects only one)
    if 'Optimizers' in config['HLSConfig'] and 'SkipOptimizers' in config['HLSConfig']:
        del config['HLSConfig']['SkipOptimizers']

    try:
        # Convert model to HLS4ML (hls4ml 1.1.0 expects only config)
        hls_model = hls4ml.converters.keras_to_hls(config)
        print(f"HLS4ML model created in: {output_dir}")
        return hls_model
    except Exception as e:
        print(f"Error converting to HLS4ML: {e}")
        raise


def validate_conversion(model, hls_model, output_dir):
    """
    Validate the HLS4ML conversion with test data

    Args:
        model: Original Keras model
        hls_model: HLS4ML model
        output_dir: Output directory for results
    """
    print("Validating model conversion...")

    # Generate test data
    print("Generating test data...")
    test_data = np.random.rand(1, INPUT_HEIGHT, INPUT_WIDTH, 3).astype(np.float32)

    # Get predictions from original model
    print("Running inference on original model...")
    start_time = time.time()
    original_pred = model.predict(test_data, verbose=0)
    original_time = time.time() - start_time

    # Get predictions from HLS4ML model
    print("Running inference on HLS4ML model...")
    start_time = time.time()
    hls_pred = hls_model.predict(test_data)
    hls_time = time.time() - start_time

    # Compare predictions
    print("Comparing predictions...")
    mse = np.mean((original_pred - hls_pred) ** 2)
    mae = np.mean(np.abs(original_pred - hls_pred))

    print(f"Validation Results:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Original model time: {original_time:.4f}s")
    print(f"  HLS4ML model time: {hls_time:.4f}s")

    # Save validation results
    validation_results = {
        "mse": float(mse),
        "mae": float(mae),
        "original_time": float(original_time),
        "hls_time": float(hls_time),
        "test_data_shape": test_data.shape,
        "original_pred_shape": original_pred.shape,
        "hls_pred_shape": hls_pred.shape,
    }

    results_file = os.path.join(output_dir, "validation_results.json")
    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"Validation results saved to: {results_file}")

    return validation_results


def generate_performance_report(hls_model, output_dir):
    """
    Generate performance and resource report

    Args:
        hls_model: HLS4ML model
        output_dir: Output directory
    """
    print("Generating performance report...")

    try:
        # Get model summary
        hls_model.summary()

        # Get resource estimates
        if hasattr(hls_model, "get_resource_usage"):
            resources = hls_model.get_resource_usage()
            print(f"Resource Usage: {resources}")

        # Save performance report
        report = {
            "model_name": hls_model.name,
            "input_shape": hls_model.input_shape,
            "output_shape": hls_model.output_shape,
            "total_params": hls_model.get_weights_count(),
            "generated_at": datetime.now().isoformat(),
            "hls4ml_version": hls4ml.__version__,
        }

        report_file = os.path.join(output_dir, "performance_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {report_file}")

    except Exception as e:
        print(f"Warning: Could not generate complete performance report: {e}")


def run_synthesis(hls_model, output_dir):
    """
    Run Vivado HLS synthesis

    Args:
        hls_model: HLS4ML model
        output_dir: Output directory
    """
    print("Running Vivado HLS synthesis...")

    try:
        # Build the project
        hls_model.build(
            csim=True,  # Run C simulation
            synth=True,  # Run synthesis
            cosim=True,  # Run co-simulation
            export=True,  # Export IP
        )

        print("Synthesis completed successfully!")

        # Check synthesis results
        if os.path.exists(
            os.path.join(output_dir, "myproject_prj", "solution1", "syn")
        ):
            print("Synthesis files generated successfully")

        if os.path.exists(
            os.path.join(output_dir, "myproject_prj", "solution1", "impl")
        ):
            print("Implementation files generated successfully")

    except Exception as e:
        print(f"Error during synthesis: {e}")
        print("Synthesis failed, but HLS4ML model was created successfully")
        print("You can manually run synthesis later using the generated project")


def create_deployment_guide(output_dir, args):
    """
    Create deployment guide for the FPGA model

    Args:
        output_dir: Output directory
        args: Command line arguments
    """
    print("Creating deployment guide...")

    guide_content = f"""
# LMIINet FPGA Deployment Guide

## Model Information
- **Model Type**: {'Quantized (QAT)' if args.quantized else 'Float32'}
- **Input Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x3
- **Output Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x{NUM_CLASSES}
- **FPGA Part**: {args.part}
- **Clock Period**: {args.clock_period}ns
- **I/O Type**: {args.io_type}
- **Strategy**: {args.strategy}

## Generated Files
The following files were generated in: {output_dir}

### Core Files
- `myproject.cpp` - Main HLS implementation
- `myproject.h` - Header file
- `myproject_test.cpp` - Testbench
- `myproject_tb.cpp` - Testbench implementation

### Project Files
- `myproject_prj/` - Vivado HLS project directory
- `myproject_prj/solution1/` - Synthesis solution
- `myproject_prj/solution1/syn/` - Synthesis reports
- `myproject_prj/solution1/impl/` - Implementation files

## Usage Instructions

### 1. Simulation
```bash
cd {output_dir}
./myproject
```

### 2. Synthesis
```bash
cd {output_dir}
vivado_hls -f run_hls.tcl
```

### 3. Co-simulation
```bash
cd {output_dir}
./myproject
```

### 4. IP Export
The synthesized IP core can be found in:
`{output_dir}/myproject_prj/solution1/impl/ip/`

## Integration with Vivado
1. Open Vivado and create a new project
2. Add the generated IP core to your project
3. Instantiate the IP core in your design
4. Connect input/output ports as needed
5. Generate bitstream

## Performance Notes
- The model is optimized for {args.strategy} strategy
- Reuse factor: {args.reuse_factor}
- Precision: {args.precision}
- Expected latency and resource usage can be found in synthesis reports

## Troubleshooting
- Check synthesis logs in `myproject_prj/solution1/syn/report/`
- Verify timing constraints are met
- Ensure sufficient FPGA resources are available
- Check I/O timing for your specific application

## Support
For issues with HLS4ML conversion, check:
- HLS4ML documentation: https://hls4ml.readthedocs.io/
- Model compatibility with supported layers
- FPGA resource constraints
- Timing requirements
"""

    guide_file = os.path.join(output_dir, "DEPLOYMENT_GUIDE.md")
    with open(guide_file, "w") as f:
        f.write(guide_content)

    print(f"Deployment guide saved to: {guide_file}")


def main():
    """Main function"""
    global args
    args = parse_arguments()

    print("=" * 60)
    print("HLS4ML FPGA Model Builder for LMIINet")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(
        args.output_dir,
        f"lmiinet_{'qat' if args.quantized else 'float32'}_{int(time.time())}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Quantized: {args.quantized}")
    print(f"FPGA part: {args.part}")
    print(f"Clock period: {args.clock_period}ns")
    print(f"I/O type: {args.io_type}")
    print(f"Strategy: {args.strategy}")
    print(f"Precision: {args.precision}")

    try:
        # Step 1: Load trained model
        model = load_trained_model(args.model_path, args.quantized)

        # Step 2: Create HLS4ML configuration
        config = create_hls4ml_config(args)

        # Step 3: Convert to HLS4ML
        hls_model = convert_to_hls4ml(model, config, output_dir)

        # Step 4: Validate conversion (if requested)
        if args.validate:
            validation_results = validate_conversion(model, hls_model, output_dir)

        # Step 5: Generate performance report
        generate_performance_report(hls_model, output_dir)

        # Step 6: Run synthesis (if requested)
        if args.synthesize:
            run_synthesis(hls_model, output_dir)

        # Step 7: Create deployment guide
        create_deployment_guide(output_dir, args)

        print("\n" + "=" * 60)
        print("✅ FPGA Model Build Completed Successfully!")
        print("=" * 60)
        print(f"📁 Output directory: {output_dir}")
        print(f"🔧 HLS4ML project: {output_dir}/myproject_prj/")
        print(f"📋 Deployment guide: {output_dir}/DEPLOYMENT_GUIDE.md")

        if args.validate:
            print(f"✅ Validation results: {output_dir}/validation_results.json")

        if args.synthesize:
            print(f"🔧 Synthesis results: {output_dir}/myproject_prj/solution1/")

        print("\nNext steps:")
        print("1. Review the deployment guide")
        print("2. Run simulation: cd {output_dir} && ./myproject")
        print("3. Run synthesis: vivado_hls -f run_hls.tcl")
        print("4. Integrate with your Vivado project")

    except Exception as e:
        print(f"\n❌ Error during FPGA model build: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
