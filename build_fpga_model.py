#!/usr/bin/env python3
"""
HLS4ML FPGA Model Builder for LMIINet Semantic Segmentation
===========================================================

This script converts a trained LMIINet QAT model to HLS4ML format and generates
FPGA implementation code. It follows hls4ml best practices and examples.

Usage:
    python build_fpga_model.py --model_path ./models/qat_model_quantized_weights.h5
    python build_fpga_model.py --model_path ./models/qat_model_clean_for_hls4ml.h5 --validate

Features:
- Loads trained LMIINet QAT models
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

Example:
    docker-compose exec hls4ml-training python3 build_fpga_model.py \
        --model_path ./models/qat_model_quantized_weights.h5 \
        --output_dir ./fpga_models/optimized_qat \
        --strategy Latency \
        --reuse_factor 2 \
        --validate
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
import pprint

# Model constants from training script
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 20


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FPGA model using HLS4ML")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained QAT model (.h5 file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fpga_models",
        help="Output directory for FPGA model",
    )
    parser.add_argument(
        "--part", 
        type=str, 
        default="xczu7ev-ffvc1156-2-e", 
        help="FPGA part number"
    )
    parser.add_argument(
        "--clock_period", 
        type=float, 
        default=5.0, 
        help="Clock period in nanoseconds"
    )
    parser.add_argument(
        "--io_type",
        type=str,
        default="io_stream",  # Changed default to io_stream for semantic segmentation
        choices=["io_parallel", "io_stream"],
        help="I/O type for HLS4ML (io_stream recommended for semantic segmentation)",
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
        "--validate",
        action="store_true",
        help="Validate model conversion with test data",
    )
    parser.add_argument(
        "--synthesize", 
        action="store_true", 
        help="Run Vivado HLS synthesis"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )

    return parser.parse_args()


def load_qat_model(model_path):
    """
    Load QAT model and prepare for HLS4ML conversion

    Args:
        model_path: Path to the QAT model (.h5 file)

    Returns:
        Loaded QAT model
    """
    print(f"Loading QAT model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Import QKeras components for loading quantized models
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

        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        
        print(f"✅ Model loaded successfully: {model.name}")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        # Verify model structure
        verify_model_structure(model)
        
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def verify_model_structure(model):
    """
    Verify that the model has the expected structure for HLS4ML

    Args:
        model: Keras model to verify
    """
    print("🔍 Verifying model structure...")

    # Check input/output shapes
    input_shape = model.input_shape[1:] if isinstance(model.input_shape, tuple) else model.input_shape[0][1:]
    output_shape = model.output_shape[1:] if isinstance(model.output_shape, tuple) else model.output_shape[0][1:]

    print(f"📥 Input shape: {input_shape}")
    print(f"📤 Output shape: {output_shape}")

    expected_input = (INPUT_HEIGHT, INPUT_WIDTH, 3)
    expected_output = (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)

    if input_shape != expected_input:
        print(f"⚠️  Warning: Expected input shape {expected_input}, got {input_shape}")
    
    if output_shape != expected_output:
        print(f"⚠️  Warning: Expected output shape {expected_output}, got {output_shape}")

    # Check for HLS4ML-compatible layers
    print("🔍 Checking layer compatibility...")
    supported_layers = [
        "InputLayer", "Conv2D", "QConv2D", "DepthwiseConv2D", "QDepthwiseConv2D",
        "BatchNormalization", "QBatchNormalization", "Activation", "QActivation",
        "Dense", "QDense", "GlobalAveragePooling2D", "Reshape", "Add", 
        "Concatenate", "UpSampling2D", "Dropout"
    ]

    unsupported_layers = []
    for layer in model.layers:
        layer_type = type(layer).__name__
        if layer_type not in supported_layers:
            unsupported_layers.append(f"{layer.name} ({layer_type})")

    if unsupported_layers:
        print("⚠️  Warning: Found potentially unsupported layers:")
        for layer in unsupported_layers:
            print(f"   - {layer}")
    else:
        print("✅ All layers are HLS4ML-compatible")

    # Count model parameters
    total_params = model.count_params()
    print(f"📊 Total parameters: {total_params:,}")


def create_hls4ml_config(args):
    """
    Create HLS4ML configuration optimized for QAT models

    Args:
        args: Command line arguments

    Returns:
        HLS4ML configuration dictionary
    """
    print("⚙️  Creating HLS4ML configuration...")

    # Base configuration following hls4ml examples
    config = {
        "Backend": "Vivado",
        "IOType": args.io_type,  # IOType at top level as per hls4ml docs
        "Model": {
            "Precision": "ap_fixed<16,6>",  # Default precision
            "ReuseFactor": args.reuse_factor,
            "Strategy": args.strategy,
        },
        "LayerType": {
            # Conv2D layers - higher precision for accuracy
            "Conv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            "QConv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            # DepthwiseConv2D layers
            "DepthwiseConv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            "QDepthwiseConv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            # BatchNormalization layers - lower precision for efficiency
            "BatchNormalization": {
                "Precision": "ap_fixed<16,4>",
            },
            "QBatchNormalization": {
                "Precision": "ap_fixed<16,4>",
            },
            # Dense layers
            "Dense": {
                "Precision": "ap_fixed<16,4>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            "QDense": {
                "Precision": "ap_fixed<16,4>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": args.strategy,
            },
            # Activation layers
            "Activation": {
                "Precision": "ap_fixed<32,16>",
            },
            "QActivation": {
                "Precision": "ap_fixed<32,16>",
            },
        },
        "Optimizers": {
            "eliminate_linear_activation": True,
            "fuse_batch_norm_quantized_tanh": True,
            "fuse_consecutive_batch_norm": True,
        },
        "LayerConfig": {},
    }

    # Add FPGA configuration
    config["FPGA"] = {
        "Part": args.part,
        "ClockPeriod": args.clock_period,
    }
    
    # IOType is set at the top level as per hls4ml documentation

    print("✅ HLS4ML configuration created")
    if args.verbose:
        print("Configuration details:")
        pprint.pprint(config)

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
    print("🔄 Converting model to HLS4ML format...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Add model to config for hls4ml conversion
        config["KerasModel"] = model
        
        # Create HLSConfig structure for hls4ml compatibility
        config["HLSConfig"] = {
            "Model": config.get("Model", {}),
            "LayerName": config.get("LayerName", {}),
            "LayerType": config.get("LayerType", {}),
            "Optimizers": config.get("Optimizers", {}),
            "LayerConfig": config.get("LayerConfig", {}),
            "FPGA": config.get("FPGA", {}),
        }
        
        # IOType is already set at the top level of the config

        # Convert model to HLS4ML
        hls_model = hls4ml.converters.keras_to_hls(config)
        
        print(f"✅ HLS4ML model created successfully")
        print(f"📁 Output directory: {output_dir}")
        
        return hls_model

    except Exception as e:
        print(f"❌ Error converting to HLS4ML: {e}")
        raise


def validate_conversion(model, hls_model, output_dir):
    """
    Validate the HLS4ML conversion with test data

    Args:
        model: Original Keras model
        hls_model: HLS4ML model
        output_dir: Output directory for results
    """
    print("🧪 Validating model conversion...")

    # Generate test data
    print("📊 Generating test data...")
    test_data = np.random.rand(1, INPUT_HEIGHT, INPUT_WIDTH, 3).astype(np.float32)

    # Get predictions from original model
    print("🔍 Running inference on original model...")
    start_time = time.time()
    original_pred = model.predict(test_data, verbose=0)
    original_time = time.time() - start_time

    # Get predictions from HLS4ML model
    print("🔍 Running inference on HLS4ML model...")
    start_time = time.time()
    hls_pred = hls_model.predict(test_data)
    hls_time = time.time() - start_time

    # Compare predictions
    print("📈 Comparing predictions...")
    mse = np.mean((original_pred - hls_pred) ** 2)
    mae = np.mean(np.abs(original_pred - hls_pred))
    max_error = np.max(np.abs(original_pred - hls_pred))

    print(f"📊 Validation Results:")
    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   Mean Absolute Error: {mae:.6f}")
    print(f"   Max Absolute Error: {max_error:.6f}")
    print(f"   Original model time: {original_time:.4f}s")
    print(f"   HLS4ML model time: {hls_time:.4f}s")

    # Save validation results
    validation_results = {
        "mse": float(mse),
        "mae": float(mae),
        "max_error": float(max_error),
        "original_time": float(original_time),
        "hls_time": float(hls_time),
        "test_data_shape": test_data.shape,
        "original_pred_shape": original_pred.shape,
        "hls_pred_shape": hls_pred.shape,
        "validation_timestamp": datetime.now().isoformat(),
    }

    results_file = os.path.join(output_dir, "validation_results.json")
    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"💾 Validation results saved to: {results_file}")

    return validation_results


def generate_performance_report(hls_model, output_dir, args):
    """
    Generate performance and resource report

    Args:
        hls_model: HLS4ML model
        output_dir: Output directory
        args: Command line arguments
    """
    print("📊 Generating performance report...")

    try:
        # Get model summary
        hls_model.summary()

        # Create performance report
        report = {
            "model_name": hls_model.name,
            "input_shape": hls_model.input_shape,
            "output_shape": hls_model.output_shape,
            "total_params": hls_model.get_weights_count(),
            "fpga_part": args.part,
            "clock_period": args.clock_period,
            "io_type": args.io_type,
            "strategy": args.strategy,
            "reuse_factor": args.reuse_factor,
            "generated_at": datetime.now().isoformat(),
            "hls4ml_version": hls4ml.__version__,
        }

        # Try to get resource estimates if available
        try:
            if hasattr(hls_model, "get_resource_usage"):
                resources = hls_model.get_resource_usage()
                report["resource_usage"] = resources
                print(f"📊 Resource Usage: {resources}")
        except Exception as e:
            print(f"⚠️  Could not get resource usage: {e}")

        report_file = os.path.join(output_dir, "performance_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"💾 Performance report saved to: {report_file}")

    except Exception as e:
        print(f"⚠️  Could not generate complete performance report: {e}")


def run_synthesis(hls_model, output_dir):
    """
    Run Vivado HLS synthesis

    Args:
        hls_model: HLS4ML model
        output_dir: Output directory
    """
    print("🔧 Running Vivado HLS synthesis...")

    try:
        # Build the project
        hls_model.build(
            csim=True,    # Run C simulation
            synth=True,   # Run synthesis
            cosim=True,   # Run co-simulation
            export=True,  # Export IP
        )

        print("✅ Synthesis completed successfully!")

        # Check synthesis results
        synth_dir = os.path.join(output_dir, "myproject_prj", "solution1", "syn")
        impl_dir = os.path.join(output_dir, "myproject_prj", "solution1", "impl")
        
        if os.path.exists(synth_dir):
            print(f"📁 Synthesis files: {synth_dir}")
        
        if os.path.exists(impl_dir):
            print(f"📁 Implementation files: {impl_dir}")

    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        print("⚠️  Synthesis failed, but HLS4ML model was created successfully")
        print("💡 You can manually run synthesis later using the generated project")


def create_deployment_guide(output_dir, args):
    """
    Create deployment guide for the FPGA model

    Args:
        output_dir: Output directory
        args: Command line arguments
    """
    print("📝 Creating deployment guide...")

    guide_content = f"""# LMIINet FPGA Deployment Guide

## Model Information
- **Model Type**: Quantized (QAT)
- **Input Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x3
- **Output Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x{NUM_CLASSES}
- **FPGA Part**: {args.part}
- **Clock Period**: {args.clock_period}ns
- **I/O Type**: {args.io_type}
- **Strategy**: {args.strategy}
- **Reuse Factor**: {args.reuse_factor}

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

    print(f"💾 Deployment guide saved to: {guide_file}")


def main():
    """Main function"""
    args = parse_arguments()

    print("=" * 60)
    print("🚀 HLS4ML FPGA Model Builder for LMIINet")
    print("=" * 60)

    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"lmiinet_qat_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Model path: {args.model_path}")
    print(f"🔧 FPGA part: {args.part}")
    print(f"⏱️  Clock period: {args.clock_period}ns")
    print(f"🔌 I/O type: {args.io_type}")
    print(f"⚡ Strategy: {args.strategy}")
    print(f"🔄 Reuse factor: {args.reuse_factor}")

    try:
        # Step 1: Load QAT model
        model = load_qat_model(args.model_path)

        # Step 2: Create HLS4ML configuration
        config = create_hls4ml_config(args)

        # Step 3: Convert to HLS4ML
        hls_model = convert_to_hls4ml(model, config, output_dir)

        # Step 4: Validate conversion (if requested)
        if args.validate:
            validation_results = validate_conversion(model, hls_model, output_dir)

        # Step 5: Generate performance report
        generate_performance_report(hls_model, output_dir, args)

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

        print("\n🚀 Next steps:")
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
