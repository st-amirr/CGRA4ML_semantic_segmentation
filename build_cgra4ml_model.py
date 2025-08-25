#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGRA4ML Model Builder for LMIINet Semantic Segmentation
=======================================================

This script converts a trained LMIINet QAT model to CGRA4ML format and generates
FPGA/ASIC implementation code. Optimized for semantic segmentation with skip connections.

Usage:
    python build_cgra4ml_model.py --model_path ./models/qat_model_quantized_weights.h5
    python build_cgra4ml_model.py --model_path ./models/qat_model_clean_for_hls4ml.h5 --validate

Features:
- Loads trained LMIINet QAT models
- Converts to CGRA4ML-compatible format
- Handles skip connections and feature fusion
- Generates FPGA/ASIC implementation
- Configurable precision and resource constraints
- Comprehensive validation and testing
- Performance analysis and resource estimation
- Custom processing elements for semantic segmentation

Requirements:
- cgra4ml (from https://github.com/KastnerRG/cgra4ml)
- tensorflow
- qkeras (for QAT models)
- verilator or vivado (for simulation/synthesis)
- C/C++ development tools

Example:
    python build_cgra4ml_model.py --model_path ./models/qat_model_clean_for_compatible.h5 --output_dir ./cgra4ml_models --validate
    python build_cgra4ml_model.py --model_path ./models/qat_model_clean_for_compatible.h5 --output_dir ./cgra4ml_models --synthesize
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time
from datetime import datetime
import pprint
import subprocess
import platform
import glob
import shutil

# Model constants from training script
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 20

# CGRA4ML specific imports (will be installed)
try:
    import cgra4ml
    from cgra4ml import DeepSoCFlow
    from cgra4ml.deepsocflow import *
except ImportError:
    print("CGRA4ML not found. Please install from: https://github.com/KastnerRG/cgra4ml")
    print("Run: pip install -e /path/to/cgra4ml")
    sys.exit(1)


def setup_cgra4ml_environment():
    """
    Setup CGRA4ML-specific environment variables and paths
    """
    print("Setting up CGRA4ML environment...")
    
    # Set CGRA4ML environment variables
    os.environ['CGRA4ML_BACKEND'] = 'Vivado'
    os.environ['CGRA4ML_PART'] = 'xc7a200tfbg676-1'
    
    # Check for required tools
    required_tools = ['verilator', 'vivado']
    found_tools = []
    
    for tool in required_tools:
        if shutil.which(tool):
            found_tools.append(tool)
        else:
            print(f"Warning: {tool} not found in PATH")
    
    if not found_tools:
        print("Warning: No simulation/synthesis tools found")
        print("Please install Verilator or Vivado for full functionality")
    
    print(f"Found tools: {found_tools}")
    print("CGRA4ML environment setup complete")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build CGRA4ML model for LMIINet semantic segmentation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained QAT model (.h5 file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cgra4ml_models",
        help="Output directory for CGRA4ML model",
    )
    parser.add_argument(
        "--part", 
        type=str, 
        default="xc7a200tfbg676-1", 
        help="FPGA part number"
    )
    parser.add_argument(
        "--clock_period", 
        type=float, 
        default=5.0, 
        help="Clock period in nanoseconds"
    )
    parser.add_argument(
        "--data_width",
        type=int,
        default=16,
        help="Data width for quantization (8, 16, 32)",
    )
    parser.add_argument(
        "--skip_connection_strategy",
        type=str,
        default="memory_buffered",
        choices=["memory_buffered", "streaming", "hybrid"],
        help="Strategy for implementing skip connections",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model conversion with test data",
    )
    parser.add_argument(
        "--synthesize", 
        action="store_true", 
        help="Run CGRA4ML synthesis"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation after synthesis"
    )

    return parser.parse_args()


def load_qat_model(model_path):
    """
    Load QAT model and prepare for CGRA4ML conversion

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
        
        print(f"Model loaded successfully: {model.name}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Verify model structure
        verify_model_structure(model)
        
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def verify_model_structure(model):
    """
    Verify that the model has the expected structure for CGRA4ML

    Args:
        model: Keras model to verify
    """
    print("Verifying model structure...")

    # Check input/output shapes
    input_shape = model.input_shape[1:] if isinstance(model.input_shape, tuple) else model.input_shape[0][1:]
    output_shape = model.output_shape[1:] if isinstance(model.output_shape, tuple) else model.output_shape[0][1:]

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

    expected_input = (INPUT_HEIGHT, INPUT_WIDTH, 3)
    expected_output = (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)

    if input_shape != expected_input:
        print(f"Warning: Expected input shape {expected_input}, got {input_shape}")
    
    if output_shape != expected_output:
        print(f"Warning: Expected output shape {expected_output}, got {output_shape}")

    # Analyze skip connections
    analyze_skip_connections(model)

    # Count model parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")


def analyze_skip_connections(model):
    """
    Analyze skip connections in the model for CGRA4ML implementation

    Args:
        model: Keras model to analyze
    """
    print("Analyzing skip connections for CGRA4ML implementation...")
    
    skip_connections = []
    layer_connections = {}
    
    # Build layer connectivity graph
    for layer in model.layers:
        layer_connections[layer.name] = []
        for node in layer._outbound_nodes:
            for outbound_layer in node.outbound_layers:
                layer_connections[layer.name].append(outbound_layer.name)
    
    # Find skip connections (layers that connect to non-adjacent layers)
    for layer in model.layers:
        for outbound_layer_name in layer_connections[layer.name]:
            # Check if this is a skip connection
            layer_idx = [i for i, l in enumerate(model.layers) if l.name == layer.name][0]
            outbound_idx = [i for i, l in enumerate(model.layers) if l.name == outbound_layer_name][0]
            
            if outbound_idx > layer_idx + 1:  # Skip connection detected
                skip_connections.append({
                    'from_layer': layer.name,
                    'to_layer': outbound_layer_name,
                    'from_idx': layer_idx,
                    'to_idx': outbound_idx,
                    'distance': outbound_idx - layer_idx
                })
    
    print(f"Found {len(skip_connections)} skip connections:")
    for conn in skip_connections:
        print(f"  {conn['from_layer']} -> {conn['to_layer']} (distance: {conn['distance']})")
    
    return skip_connections


def create_cgra4ml_config(args, skip_connections):
    """
    Create CGRA4ML configuration optimized for LMIINet with skip connections

    Args:
        args: Command line arguments
        skip_connections: List of skip connections found in the model

    Returns:
        CGRA4ML configuration dictionary
    """
    print("Creating CGRA4ML configuration...")

    # Base configuration for CGRA4ML
    config = {
        "model_name": "lmiinet_semantic_segmentation",
        "input_shape": [INPUT_HEIGHT, INPUT_WIDTH, 3],
        "output_shape": [INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES],
        "data_width": args.data_width,
        "clock_period": args.clock_period,
        "fpga_part": args.part,
        
        # Skip connection configuration
        "skip_connections": {
            "strategy": args.skip_connection_strategy,
            "connections": skip_connections,
            "memory_buffers": {},
            "streaming_config": {
                "buffer_size": 1024,
                "fifo_depth": 512
            }
        },
        
        # Layer configurations
        "layers": {
            "conv2d": {
                "precision": f"ap_fixed<{args.data_width},8>",
                "strategy": "resource",
                "reuse_factor": 1
            },
            "depthwise_conv2d": {
                "precision": f"ap_fixed<{args.data_width},8>",
                "strategy": "resource",
                "reuse_factor": 1
            },
            "batch_norm": {
                "precision": f"ap_fixed<{args.data_width},4>",
                "strategy": "latency"
            },
            "activation": {
                "precision": f"ap_fixed<{args.data_width},8>",
                "strategy": "latency"
            },
            "upsampling": {
                "precision": f"ap_fixed<{args.data_width},8>",
                "strategy": "resource",
                "interpolation": "nearest"
            },
            "dense": {
                "precision": f"ap_fixed<{args.data_width},4>",
                "strategy": "resource",
                "reuse_factor": 1
            }
        },
        
        # Memory configuration
        "memory": {
            "bram_usage": "optimize",
            "array_partition": True,
            "partition_factor": 2,
            "buffer_strategy": "double_buffering"
        },
        
        # Performance configuration
        "performance": {
            "pipeline_depth": 10,
            "parallel_factor": 4,
            "dataflow": True,
            "optimization_level": 3
        },
        
        # Custom processing elements for semantic segmentation
        "custom_pe": {
            "skip_connection_pe": {
                "type": "memory_controller",
                "buffer_size": 2048,
                "data_width": args.data_width
            },
            "feature_fusion_pe": {
                "type": "arithmetic",
                "operation": "add",
                "data_width": args.data_width
            },
            "upsampling_pe": {
                "type": "interpolation",
                "method": "nearest_neighbor",
                "data_width": args.data_width
            }
        }
    }

    print("CGRA4ML configuration created")
    if args.verbose:
        print("Configuration details:")
        pprint.pprint(config)

    return config


def create_custom_processing_elements(config):
    """
    Create custom processing elements for semantic segmentation

    Args:
        config: CGRA4ML configuration

    Returns:
        Dictionary of custom processing elements
    """
    print("Creating custom processing elements for semantic segmentation...")
    
    custom_pe = {}
    
    # Skip Connection Processing Element
    skip_pe = {
        "name": "skip_connection_pe",
        "type": "memory_controller",
        "description": "Handles skip connections with memory buffering",
        "ports": {
            "input": {"width": config["data_width"], "type": "stream"},
            "output": {"width": config["data_width"], "type": "stream"},
            "control": {"width": 32, "type": "control"}
        },
        "parameters": {
            "buffer_size": config["skip_connections"]["streaming_config"]["buffer_size"],
            "data_width": config["data_width"]
        }
    }
    custom_pe["skip_connection_pe"] = skip_pe
    
    # Feature Fusion Processing Element
    fusion_pe = {
        "name": "feature_fusion_pe",
        "type": "arithmetic",
        "description": "Performs feature fusion for skip connections",
        "ports": {
            "input1": {"width": config["data_width"], "type": "stream"},
            "input2": {"width": config["data_width"], "type": "stream"},
            "output": {"width": config["data_width"], "type": "stream"},
            "control": {"width": 32, "type": "control"}
        },
        "parameters": {
            "operation": "add",
            "data_width": config["data_width"]
        }
    }
    custom_pe["feature_fusion_pe"] = fusion_pe
    
    # Upsampling Processing Element
    upsampling_pe = {
        "name": "upsampling_pe",
        "type": "interpolation",
        "description": "Performs nearest neighbor upsampling",
        "ports": {
            "input": {"width": config["data_width"], "type": "stream"},
            "output": {"width": config["data_width"], "type": "stream"},
            "control": {"width": 32, "type": "control"}
        },
        "parameters": {
            "method": "nearest_neighbor",
            "scale_factor": 2,
            "data_width": config["data_width"]
        }
    }
    custom_pe["upsampling_pe"] = upsampling_pe
    
    print(f"Created {len(custom_pe)} custom processing elements")
    return custom_pe


def convert_to_cgra4ml(model, config, output_dir):
    """
    Convert Keras model to CGRA4ML format

    Args:
        model: Keras model
        config: CGRA4ML configuration
        output_dir: Output directory

    Returns:
        CGRA4ML model
    """
    print("Converting model to CGRA4ML format...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create custom processing elements
        custom_pe = create_custom_processing_elements(config)
        
        # Initialize CGRA4ML DeepSoCFlow
        print("Initializing CGRA4ML DeepSoCFlow...")
        cgra_model = DeepSoCFlow(
            model=model,
            config=config,
            custom_pe=custom_pe,
            output_dir=output_dir
        )
        
        # Generate CGRA4ML implementation
        print("Generating CGRA4ML implementation...")
        cgra_model.generate()
        
        print(f"CGRA4ML model created successfully")
        print(f"Output directory: {output_dir}")
        
        # Write project files
        print("Writing CGRA4ML project files...")
        try:
            cgra_model.write_project_files()
            print("CGRA4ML project files written successfully")
            
        except Exception as e:
            print(f"Warning: Could not write all project files: {e}")
            print("CGRA4ML model was created successfully, but project files may be incomplete")
        
        return cgra_model

    except Exception as e:
        print(f"Error converting to CGRA4ML: {e}")
        raise


def run_cgra4ml_synthesis(output_dir, args):
    """
    Run CGRA4ML synthesis on the generated project
    
    Args:
        output_dir: Output directory containing the CGRA4ML project
        args: Command line arguments
    """
    print("=" * 60)
    print("Running CGRA4ML Synthesis")
    print("=" * 60)
    
    # Change to the project directory
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        print(f"Changed to directory: {output_dir}")
        
        # Check if synthesis script exists
        synthesis_script = "run_synthesis.tcl"
        if not os.path.exists(synthesis_script):
            print(f"ERROR: {synthesis_script} not found!")
            return False
        
        # Run synthesis
        print("Starting CGRA4ML synthesis...")
        print(f"Command: vivado -mode batch -source {synthesis_script}")
        
        # Run the synthesis command
        result = subprocess.run(["vivado", "-mode", "batch", "-source", synthesis_script], 
                              capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("Synthesis Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Synthesis Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("=" * 60)
            print("CGRA4ML synthesis completed successfully!")
            print("=" * 60)
            
            # Check for synthesis reports
            syn_report_dir = "synthesis_reports"
            if os.path.exists(syn_report_dir):
                print(f"Synthesis reports available in: {syn_report_dir}")
                
                # List available reports
                reports = [f for f in os.listdir(syn_report_dir) if f.endswith('.rpt')]
                if reports:
                    print("Available reports:")
                    for report in reports:
                        print(f"  - {report}")
            
            return True
        else:
            print("=" * 60)
            print("CGRA4ML synthesis failed!")
            print("=" * 60)
            print(f"Return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir(original_dir)
        print(f"Changed back to directory: {original_dir}")


def run_cgra4ml_simulation(output_dir, args):
    """
    Run CGRA4ML simulation
    
    Args:
        output_dir: Output directory containing the CGRA4ML project
        args: Command line arguments
    """
    print("=" * 60)
    print("Running CGRA4ML Simulation")
    print("=" * 60)
    
    # Change to the project directory
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        print(f"Changed to directory: {output_dir}")
        
        # Check if simulation script exists
        sim_script = "run_simulation.sh"
        if not os.path.exists(sim_script):
            print(f"ERROR: {sim_script} not found!")
            return False
        
        # Run simulation
        print("Starting CGRA4ML simulation...")
        print(f"Command: bash {sim_script}")
        
        # Run the simulation command
        result = subprocess.run(["bash", sim_script], 
                              capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("Simulation Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Simulation Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("=" * 60)
            print("CGRA4ML simulation completed successfully!")
            print("=" * 60)
            return True
        else:
            print("=" * 60)
            print("CGRA4ML simulation failed!")
            print("=" * 60)
            print(f"Return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error during simulation: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir(original_dir)
        print(f"Changed back to directory: {original_dir}")


def validate_conversion(model, cgra_model, output_dir):
    """
    Validate the CGRA4ML conversion with test data

    Args:
        model: Original Keras model
        cgra_model: CGRA4ML model
        output_dir: Output directory for results
    """
    print("Validating CGRA4ML model conversion...")

    # Generate test data
    print("Generating test data...")
    test_data = np.random.rand(1, INPUT_HEIGHT, INPUT_WIDTH, 3).astype(np.float32)

    # Get predictions from original model
    print("Running inference on original model...")
    start_time = time.time()
    original_pred = model.predict(test_data, verbose=0)
    original_time = time.time() - start_time

    # Get predictions from CGRA4ML model
    print("Running inference on CGRA4ML model...")
    start_time = time.time()
    cgra_pred = cgra_model.predict(test_data)
    cgra_time = time.time() - start_time

    # Compare predictions
    print("Comparing predictions...")
    print(f"   Original prediction shape: {original_pred.shape}")
    print(f"   CGRA4ML prediction shape: {cgra_pred.shape}")
    
    # Reshape CGRA4ML predictions to match original model output
    if cgra_pred.ndim == 1:
        cgra_pred = cgra_pred.reshape(original_pred.shape)
        print(f"   Reshaped CGRA4ML prediction to: {cgra_pred.shape}")
    
    mse = np.mean((original_pred - cgra_pred) ** 2)
    mae = np.mean(np.abs(original_pred - cgra_pred))
    max_error = np.max(np.abs(original_pred - cgra_pred))

    print(f"Validation Results:")
    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   Mean Absolute Error: {mae:.6f}")
    print(f"   Max Absolute Error: {max_error:.6f}")
    print(f"   Original model time: {original_time:.4f}s")
    print(f"   CGRA4ML model time: {cgra_time:.4f}s")

    # Save validation results
    validation_results = {
        "mse": float(mse),
        "mae": float(mae),
        "max_error": float(max_error),
        "original_time": float(original_time),
        "cgra_time": float(cgra_time),
        "test_data_shape": test_data.shape,
        "original_pred_shape": original_pred.shape,
        "cgra_pred_shape": cgra_pred.shape,
        "validation_timestamp": datetime.now().isoformat(),
    }

    results_file = os.path.join(output_dir, "validation_results.json")
    with open(results_file, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"Validation results saved to: {results_file}")

    return validation_results


def generate_performance_report(cgra_model, output_dir, args):
    """
    Generate performance and resource report

    Args:
        cgra_model: CGRA4ML model
        output_dir: Output directory
        args: Command line arguments
    """
    print("Generating CGRA4ML performance report...")

    try:
        # Create performance report
        report = {
            "model_name": "lmiinet_semantic_segmentation",
            "framework": "CGRA4ML",
            "input_shape": [INPUT_HEIGHT, INPUT_WIDTH, 3],
            "output_shape": [INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES],
            "data_width": args.data_width,
            "fpga_part": args.part,
            "clock_period": args.clock_period,
            "skip_connection_strategy": args.skip_connection_strategy,
            "generated_at": datetime.now().isoformat(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }
        
        # Try to get model summary if available
        try:
            if hasattr(cgra_model, 'summary'):
                cgra_model.summary()
                print("Model summary generated")
            else:
                print("Model summary not available for CGRA4ML model")
        except Exception as e:
            print(f"Could not generate model summary: {e}")

        # Try to get resource estimates if available
        try:
            if hasattr(cgra_model, "get_resource_usage"):
                resources = cgra_model.get_resource_usage()
                report["resource_usage"] = resources
                print(f"Resource Usage: {resources}")
        except Exception as e:
            print(f"Could not get resource usage: {e}")

        report_file = os.path.join(output_dir, "cgra4ml_performance_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"CGRA4ML performance report saved to: {report_file}")

    except Exception as e:
        print(f"Could not generate complete performance report: {e}")


def create_deployment_guide(output_dir, args):
    """
    Create deployment guide for the CGRA4ML model

    Args:
        output_dir: Output directory
        args: Command line arguments
    """
    print("Creating CGRA4ML deployment guide...")

    guide_content = f"""# LMIINet CGRA4ML Deployment Guide

## Model Information
- **Model Type**: LMIINet Semantic Segmentation (QAT)
- **Framework**: CGRA4ML
- **Input Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x3
- **Output Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x{NUM_CLASSES}
- **FPGA Part**: {args.part}
- **Clock Period**: {args.clock_period}ns
- **Data Width**: {args.data_width} bits
- **Skip Connection Strategy**: {args.skip_connection_strategy}

## CGRA4ML Advantages
- **Better Skip Connection Support**: Optimized for semantic segmentation
- **Single Reconfigurable Engine**: More efficient than layer-per-engine approach
- **Custom Processing Elements**: Tailored for semantic segmentation operations
- **Memory-Optimized**: Efficient handling of large feature maps

## Generated Files
The following files were generated in: {output_dir}

### Core Files
- `lmiinet_semantic_segmentation.cpp` - Main CGRA4ML implementation
- `lmiinet_semantic_segmentation.h` - Header file
- `custom_pe/` - Custom processing elements
- `skip_connection_pe.v` - Skip connection processing element
- `feature_fusion_pe.v` - Feature fusion processing element
- `upsampling_pe.v` - Upsampling processing element

### Project Files
- `run_synthesis.tcl` - Vivado synthesis script
- `run_simulation.sh` - Simulation script
- `constraints.xdc` - Timing constraints
- `synthesis_reports/` - Synthesis reports

## Usage Instructions

### 1. Simulation
```bash
cd {output_dir}
bash run_simulation.sh
```

### 2. Synthesis
```bash
cd {output_dir}
vivado -mode batch -source run_synthesis.tcl
```

### 3. Integration
The synthesized IP core can be found in:
`{output_dir}/synthesis_reports/`

## Skip Connection Implementation
CGRA4ML implements skip connections using:
- **Memory Buffering**: Efficient storage of intermediate features
- **Custom Processing Elements**: Specialized for feature fusion
- **Streaming Architecture**: Optimized data flow

## Performance Notes
- Optimized for semantic segmentation workloads
- Efficient memory usage for large feature maps
- Custom processing elements for skip connections
- Configurable precision and resource usage

## Troubleshooting
- Check synthesis logs in `synthesis_reports/`
- Verify timing constraints are met
- Ensure sufficient FPGA resources
- Check skip connection memory requirements

## Support
For issues with CGRA4ML:
- CGRA4ML Repository: https://github.com/KastnerRG/cgra4ml
- Check custom processing element implementations
- Verify skip connection configurations
- Review memory buffer sizing
"""

    guide_file = os.path.join(output_dir, "CGRA4ML_DEPLOYMENT_GUIDE.md")
    with open(guide_file, "w") as f:
        f.write(guide_content)

    print(f"CGRA4ML deployment guide saved to: {guide_file}")


def main():
    """Main function"""
    args = parse_arguments()

    print("=" * 60)
    print("CGRA4ML Model Builder for LMIINet Semantic Segmentation")
    print("=" * 60)

    # Setup CGRA4ML environment
    setup_cgra4ml_environment()

    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"lmiinet_cgra4ml_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"FPGA part: {args.part}")
    print(f"Clock period: {args.clock_period}ns")
    print(f"Data width: {args.data_width} bits")
    print(f"Skip connection strategy: {args.skip_connection_strategy}")
    print(f"Platform: {platform.system()}")

    try:
        # Step 1: Load QAT model
        model = load_qat_model(args.model_path)

        # Step 2: Analyze skip connections
        skip_connections = analyze_skip_connections(model)

        # Step 3: Create CGRA4ML configuration
        config = create_cgra4ml_config(args, skip_connections)

        # Step 4: Convert to CGRA4ML
        cgra_model = convert_to_cgra4ml(model, config, output_dir)

        # Step 5: Validate conversion (if requested)
        validation_results = None
        if args.validate:
            validation_results = validate_conversion(model, cgra_model, output_dir)

        # Step 6: Generate performance report
        generate_performance_report(cgra_model, output_dir, args)

        # Step 7: Create deployment guide
        create_deployment_guide(output_dir, args)
        
        # Step 8: Run synthesis if requested
        if args.synthesize:
            run_cgra4ml_synthesis(output_dir, args)
        
        # Step 9: Run simulation if requested
        if args.simulate:
            run_cgra4ml_simulation(output_dir, args)
        
        print("\n" + "=" * 60)
        print("CGRA4ML Model Build Completed Successfully!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"CGRA4ML project: {output_dir}/")
        print(f"Deployment guide: {output_dir}/CGRA4ML_DEPLOYMENT_GUIDE.md")

        if args.validate and validation_results is not None:
            print(f"Validation results: {output_dir}/validation_results.json")
        elif args.validate:
            print("Validation completed")

        print("\nNext steps:")
        print("1. Review the CGRA4ML deployment guide")
        print("2. Run simulation: cd {output_dir} && bash run_simulation.sh")
        if args.synthesize:
            print("3. Synthesis completed automatically!")
            print("4. Check synthesis reports in synthesis_reports/")
            print("5. Integrate with your Vivado project")
        else:
            print("3. Run synthesis: vivado -mode batch -source run_synthesis.tcl")
            print("4. Integrate with your Vivado project")

    except Exception as e:
        print(f"\nError during CGRA4ML model build: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
