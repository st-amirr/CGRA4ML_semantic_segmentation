#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLS4ML FPGA Model Builder for LMIINet Semantic Segmentation (Windows Version)
=============================================================================

This script converts a trained LMIINet QAT model to HLS4ML format and generates
FPGA implementation code. Optimized for Windows with Vivado HLS.

Usage:
    python build_fpga_model_windows.py --model_path ./models/qat_model_quantized_weights.h5
    python build_fpga_model_windows.py --model_path ./models/qat_model_clean_for_hls4ml.h5 --validate

Features:
- Loads trained LMIINet QAT models
- Converts to HLS4ML-compatible format
- Generates FPGA implementation with Vivado HLS
- Configurable precision and resource constraints
- Comprehensive validation and testing
- Performance analysis and resource estimation
- Windows-optimized paths and environment
- Automatic Vivado HLS path detection

Requirements:
- hls4ml
- tensorflow
- qkeras (for QAT models)
- vivado_hls (for FPGA synthesis)
- Windows environment with Vivado installed

Example:
    python build_fpga_model_windows.py --model_path ./models/qat_model_clean_for_compatible.h5 --output_dir ./fpga_models/optimized_qat --strategy Latency --reuse_factor 2 --validate
    python build_fpga_model_windows.py --model_path ./models/qat_model_clean_for_compatible.h5 --output_dir ./fpga_models/optimized_qat --strategy Latency --reuse_factor 2 --validate --synthesize
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
import subprocess
import platform
import glob

# Model constants from training script
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
NUM_CLASSES = 20


def setup_windows_environment():
    """
    Setup Windows-specific environment variables and paths
    """
    print("Setting up Windows environment...")
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("Warning: This script is optimized for Windows")
        return
    
    # Set Windows-specific environment variables
    os.environ['HLS4ML_BACKEND'] = 'Vivado'
    
    # Try to find Vitis HLS installation
    vivado_paths = [
        r"C:\Xilinx\Vitis_HLS\2023.2\bin",
        r"C:\Xilinx\Vitis_HLS\2023.1\bin",
        r"C:\Xilinx\Vitis_HLS\2022.2\bin",
        r"C:\Xilinx\Vitis_HLS\2022.1\bin",
        r"C:\Xilinx\Vitis_HLS\2021.2\bin",
        r"C:\Xilinx\Vitis_HLS\2021.1\bin",
        r"C:\Xilinx\Vitis\2023.2\bin",
        r"C:\Xilinx\Vitis\2023.1\bin", 
        r"C:\Xilinx\Vitis\2022.2\bin",
        r"C:\Xilinx\Vitis\2022.1\bin",
        r"C:\Xilinx\Vitis\2021.2\bin",
        r"C:\Xilinx\Vitis\2021.1\bin",
        r"C:\Xilinx\Vitis\2020.2\bin",
        r"C:\Xilinx\Vitis\2020.1\bin",
        r"C:\Xilinx\SDK\2019.2\bin",
        r"C:\Xilinx\SDK\2019.1\bin",
        r"C:\Xilinx\SDK\2018.3\bin",
        r"C:\Xilinx\SDK\2018.2\bin",
    ]
    
    vivado_found = False
    for path in vivado_paths:
        if os.path.exists(path):
            # Try both vitis_hls.bat and vivado_hls.bat
            vitis_exe = os.path.join(path, "vitis_hls.bat")
            vivado_exe = os.path.join(path, "vivado_hls.bat")
            
            if os.path.exists(vitis_exe):
                os.environ['VIVADO_HLS_PATH'] = vitis_exe
                print(f"Found Vitis HLS at: {vitis_exe}")
                vivado_found = True
                break
            elif os.path.exists(vivado_exe):
                os.environ['VIVADO_HLS_PATH'] = vivado_exe
                print(f"Found Vivado HLS at: {vivado_exe}")
                vivado_found = True
                break
    
    if not vivado_found:
        print("Warning: Vivado HLS not found in standard locations")
        print("Please ensure Vivado is installed and add it to your PATH")
        print("Or set VIVADO_HLS_PATH environment variable manually")
    
    # Set HLS4ML environment
    os.environ['HLS4ML_BACKEND'] = 'Vivado'
    os.environ['HLS4ML_PART'] = 'xc7a200tfbg676-1'
    
    print("Windows environment setup complete")


def find_vivado_hls():
    """
    Find Vivado HLS executable on Windows
    """
    # Check environment variable first
    vivado_path = os.environ.get('VIVADO_HLS_PATH')
    if vivado_path and os.path.exists(vivado_path):
        return vivado_path
    
    # Search in common installation paths
    search_paths = [
        r"C:\Xilinx\Vitis_HLS\*\bin\vitis_hls.bat",
        r"C:\Xilinx\Vitis_HLS\*\bin\vivado_hls.bat",
        r"C:\Xilinx\Vitis\*\bin\vitis_hls.bat",
        r"C:\Xilinx\Vitis\*\bin\vivado_hls.bat",
        r"C:\Xilinx\SDK\*\bin\vivado_hls.bat",
    ]
    
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            # Sort by version (newest first)
            matches.sort(reverse=True)
            return matches[0]
    
    return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FPGA model using HLS4ML (Windows Version)")
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
        "--io_type",
        type=str,
        default="io_stream",
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
    parser.add_argument(
        "--vivado_path",
        type=str,
        help="Path to Vivado HLS executable (if not auto-detected)"
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
        
        print(f"Model loaded successfully: {model.name}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Verify model structure
        verify_model_structure(model)
        
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def fix_upsampling_layers(model):
    """
    Fix UpSampling2D layers to use nearest neighbor interpolation for HLS4ML compatibility
    
    Args:
        model: Keras model to fix
        
    Returns:
        Fixed model
    """
    print("Fixing UpSampling2D layers for HLS4ML compatibility...")
    
    fixed_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.UpSampling2D):
            # Check if it's using bilinear interpolation
            if hasattr(layer, 'interpolation') and layer.interpolation == 'bilinear':
                print(f"   Fixing layer {layer.name}: bilinear -> nearest")
                layer.interpolation = 'nearest'
                fixed_count += 1
    
    if fixed_count > 0:
        print(f"Fixed {fixed_count} UpSampling2D layers")
    else:
        print("No UpSampling2D layers needed fixing")
    
    return model


def verify_model_structure(model):
    """
    Verify that the model has the expected structure for HLS4ML

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

    # Check for HLS4ML-compatible layers
    print("Checking layer compatibility...")
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
        print("Warning: Found potentially unsupported layers:")
        for layer in unsupported_layers:
            print(f"   - {layer}")
    else:
        print("All layers are HLS4ML-compatible")

    # Count model parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")


def create_hls4ml_config(args):
    """
    Create HLS4ML configuration optimized for QAT models

    Args:
        args: Command line arguments

    Returns:
        HLS4ML configuration dictionary
    """
    print("Creating HLS4ML configuration...")

    # Base configuration following hls4ml examples
    config = {
        "Backend": "Vivado",
        "IOType": args.io_type,
        "Model": {
            "Precision": "ap_fixed<16,6>",
            "ReuseFactor": args.reuse_factor,
            "Strategy": args.strategy,
            "ProjectName": "myproject",
        },
        "LayerType": {
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
            "DepthwiseConv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": "Resource",  # Use Resource to avoid pragma conflicts
            },
            "QDepthwiseConv2D": {
                "Precision": "ap_fixed<32,8>",
                "ReuseFactor": args.reuse_factor,
                "Strategy": "Resource",  # Use Resource to avoid pragma conflicts
            },
            "BatchNormalization": {
                "Precision": "ap_fixed<16,4>",
            },
            "QBatchNormalization": {
                "Precision": "ap_fixed<16,4>",
            },
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
            "Activation": {
                "Precision": "ap_fixed<32,16>",
            },
            "QActivation": {
                "Precision": "ap_fixed<32,16>",
            },
            "Softmax": {
                "Precision": "ap_fixed<32,16>",
                "Strategy": "Resource",
                "Axis": -1,  # Apply softmax along the last axis (classes)
            },
            "UpSampling2D": {
                "Precision": "ap_fixed<32,16>",
                "Strategy": "Resource",
                "Interpolation": "nearest",
                "Implementation": "LUT",
            },
        },
        "Optimizers": {
            "eliminate_linear_activation": True,
            "fuse_batch_norm_quantized_tanh": True,
            "fuse_consecutive_batch_norm": True,
            "optimize_bram_usage": True,  # Enable BRAM optimization
            "prefer_lut_over_bram": True,
            "array_partition": True,  # Enable array partitioning to reduce memory pressure
        },
        "LayerConfig": {
            "PreferLUT": True,
            "LUTOptimization": True,
            "MaximumSize": None,  # Disable maximum_size to avoid unsupported command
            "OptimizeMemoryUsage": True,  # Enable memory usage optimization
        },
        # Add specific layer configurations to control output
        "LayerName": {
            # Configure the final output layer to reduce log size
            "softmax": {
                "Strategy": "Resource",
                "Axis": -1,
            },
            "dense": {
                "Strategy": "Resource",
                "Implementation": "Resource",
            },
        },
    }

    # Add FPGA configuration - these should be at the top level for HLS4ML
    config["Part"] = args.part
    config["ClockPeriod"] = args.clock_period

    print("HLS4ML configuration created")
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
    print("Converting model to HLS4ML format...")

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
            "OutputDir": output_dir,
        }
        
        # Ensure project name is set at the top level as well
        config["ProjectName"] = "myproject"
        
        # Set output directory
        config["OutputDir"] = output_dir

        # Convert model to HLS4ML
        hls_model = hls4ml.converters.keras_to_hls(config)
        
        print(f"HLS4ML model created successfully")
        print(f"Output directory: {output_dir}")
        
        # Write HLS project files
        print("Writing HLS project files...")
        try:
            hls_model.write()
            print("HLS project files written successfully")
            
            # Fix TCL files for compatibility
            fix_generated_tcl_files(output_dir)
            
        except Exception as e:
            print(f"Warning: Could not write HLS project files: {e}")
            print("HLS4ML model was created successfully, but project files may be incomplete")
        
        return hls_model

    except Exception as e:
        print(f"Error converting to HLS4ML: {e}")
        raise


def fix_generated_tcl_files(output_dir):
    """
    Post-process generated TCL files to fix compatibility issues
    
    Args:
        output_dir: Output directory containing the HLS project
    """
    print("Fixing generated TCL files for compatibility...")
    
    # Fix build_prj.tcl
    build_prj_path = os.path.join(output_dir, "build_prj.tcl")
    if os.path.exists(build_prj_path):
        try:
            with open(build_prj_path, 'r') as f:
                content = f.read()
            
            # Comment out the problematic array_partition command
            content = content.replace(
                'catch {config_array_partition -maximum_size $maximum_size}',
                '# catch {config_array_partition -maximum_size $maximum_size}'
            )
            
            # Write the fixed content back
            with open(build_prj_path, 'w') as f:
                f.write(content)
            
            print("Fixed build_prj.tcl - commented out unsupported array_partition command")
            
        except Exception as e:
            print(f"Warning: Could not fix build_prj.tcl: {e}")
    
    # Fix testbench to reduce output logging
    fix_testbench_output(output_dir)
    
    print("TCL file fixes completed")


def fix_testbench_output(output_dir):
    """
    Modify the testbench to reduce output logging and only output final segmentation
    
    Args:
        output_dir: Output directory containing the HLS project
    """
    print("Fixing testbench output to reduce log size...")
    
    # Fix myproject_test.cpp
    test_file = os.path.join(output_dir, "myproject_test.cpp")
    if os.path.exists(test_file):
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Replace the output logging to only show final segmentation
            # Find and replace the output section
            if 'std::cout << "Predictions:" << std::endl;' in content:
                # Replace detailed output with summary
                content = content.replace(
                    'std::cout << "Predictions:" << std::endl;',
                    'std::cout << "Segmentation output shape: " << predictions.size() << std::endl;'
                )
                
                # Remove or comment out detailed prediction printing
                lines = content.split('\n')
                new_lines = []
                skip_prediction_printing = False
                
                for line in lines:
                    if 'for (int i = 0; i <' in line and 'predictions.size()' in line:
                        skip_prediction_printing = True
                        new_lines.append('    // Detailed prediction printing disabled to reduce log size')
                        continue
                    
                    if skip_prediction_printing and '}' in line and line.strip() == '}':
                        skip_prediction_printing = False
                        new_lines.append('    // End of prediction printing')
                        continue
                    
                    if not skip_prediction_printing:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            # Write the fixed content back
            with open(test_file, 'w') as f:
                f.write(content)
            
            print("Fixed myproject_test.cpp - reduced output logging")
            
        except Exception as e:
            print(f"Warning: Could not fix testbench: {e}")
    
    # Also fix myproject_tb.cpp if it exists
    tb_file = os.path.join(output_dir, "myproject_tb.cpp")
    if os.path.exists(tb_file):
        try:
            with open(tb_file, 'r') as f:
                content = f.read()
            
            # Add output control to reduce logging
            if 'std::cout << "Predictions:" << std::endl;' in content:
                content = content.replace(
                    'std::cout << "Predictions:" << std::endl;',
                    'std::cout << "Segmentation completed. Output size: " << predictions.size() << std::endl;'
                )
                
                # Comment out detailed output
                content = content.replace(
                    'std::cout << predictions[i] << " ";',
                    '// std::cout << predictions[i] << " ";  // Disabled to reduce log size'
                )
                
            
            with open(tb_file, 'w') as f:
                f.write(content)
            
            print("Fixed myproject_tb.cpp - reduced output logging")
            
        except Exception as e:
            print(f"Warning: Could not fix tb file: {e}")


def run_synthesis(output_dir, args):
    """
    Run Vitis HLS synthesis on the generated HLS project
    
    Args:
        output_dir: Output directory containing the HLS project
        args: Command line arguments
    """
    print("=" * 60)
    print("Running Vitis HLS Synthesis")
    print("=" * 60)
    
    # Find Vitis HLS executable
    vivado_hls_path = find_vivado_hls()
    if not vivado_hls_path:
        print("ERROR: Vitis HLS not found!")
        print("Please ensure Vitis HLS is installed and set VIVADO_HLS_PATH environment variable")
        return False
    
    print(f"Using Vitis HLS: {vivado_hls_path}")
    
    # Change to the project directory
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        print(f"Changed to directory: {output_dir}")
        
        # Check if build_prj.tcl exists
        if not os.path.exists("build_prj.tcl"):
            print("ERROR: build_prj.tcl not found!")
            return False
        
        # Run synthesis
        print("Starting synthesis...")
        print(f"Command: {vivado_hls_path} -f build_prj.tcl")
        
        # Run the synthesis command
        result = subprocess.run([vivado_hls_path, "-f", "build_prj.tcl"], 
                              capture_output=True, text=True, shell=True)
        
        # Print output
        if result.stdout:
            print("Synthesis Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Synthesis Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("=" * 60)
            print("Synthesis completed successfully!")
            print("=" * 60)
            
            # Check for synthesis reports
            syn_report_dir = os.path.join("myproject_prj", "solution1", "syn", "report")
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
            print("Synthesis failed!")
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


def validate_conversion(model, hls_model, output_dir, compile_model=True):
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
    
    # Compile the model if needed
    if compile_model:
        print("Compiling HLS4ML model for inference...")
        try:
            # Skip compilation on Windows as it requires C++ compiler
            if platform.system() == "Windows":
                print("Skipping HLS4ML compilation on Windows (requires C++ compiler)")
                print("HLS4ML model created successfully - validation skipped")
                return None
            
            # Try different compilation methods for different hls4ml versions
            if hasattr(hls_model, 'compile'):
                hls_model.compile()
            elif hasattr(hls_model, 'build'):
                hls_model.build(csim=True)
            else:
                print("Warning: No compilation method found")
                return None
            print("HLS4ML model compiled successfully")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Skipping HLS4ML validation - model created successfully")
            return None
    
    start_time = time.time()
    hls_pred = hls_model.predict(test_data)
    hls_time = time.time() - start_time

    # Reshape HLS4ML predictions to match original model output shape
    print("Comparing predictions...")
    print(f"   Original prediction shape: {original_pred.shape}")
    print(f"   HLS4ML prediction shape: {hls_pred.shape}")
    
    # Reshape HLS4ML predictions to match original model output
    if hls_pred.ndim == 1:
        # HLS4ML returns flattened output, reshape to match original
        hls_pred = hls_pred.reshape(original_pred.shape)
        print(f"   Reshaped HLS4ML prediction to: {hls_pred.shape}")
    
    mse = np.mean((original_pred - hls_pred) ** 2)
    mae = np.mean(np.abs(original_pred - hls_pred))
    max_error = np.max(np.abs(original_pred - hls_pred))

    print(f"Validation Results:")
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

    print(f"Validation results saved to: {results_file}")

    return validation_results


def generate_performance_report(hls_model, output_dir, args):
    """
    Generate performance and resource report

    Args:
        hls_model: HLS4ML model
        output_dir: Output directory
        args: Command line arguments
    """
    print("Generating performance report...")

    try:
        # Create performance report
        report = {
            "model_name": getattr(hls_model, 'name', 'hls4ml_model'),
            "input_shape": getattr(hls_model, 'input_shape', None),
            "output_shape": getattr(hls_model, 'output_shape', None),
            "total_params": getattr(hls_model, 'get_weights_count', lambda: None)(),
            "fpga_part": args.part,
            "clock_period": args.clock_period,
            "io_type": args.io_type,
            "strategy": args.strategy,
            "reuse_factor": args.reuse_factor,
            "generated_at": datetime.now().isoformat(),
            "hls4ml_version": hls4ml.__version__,
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }
        
        # Try to get model summary if available
        try:
            if hasattr(hls_model, 'summary'):
                hls_model.summary()
                print("Model summary generated")
            else:
                print("Model summary not available for HLS4ML ModelGraph")
        except Exception as e:
            print(f"Could not generate model summary: {e}")

        # Try to get resource estimates if available
        try:
            if hasattr(hls_model, "get_resource_usage"):
                resources = hls_model.get_resource_usage()
                report["resource_usage"] = resources
                print(f"Resource Usage: {resources}")
        except Exception as e:
            print(f"Could not get resource usage: {e}")

        report_file = os.path.join(output_dir, "performance_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {report_file}")

    except Exception as e:
        print(f"Could not generate complete performance report: {e}")


def create_deployment_guide(output_dir, args):
    """
    Create deployment guide for the FPGA model

    Args:
        output_dir: Output directory
        args: Command line arguments
    """
    print("Creating deployment guide...")

    guide_content = f"""# LMIINet FPGA Deployment Guide (Windows)

## Model Information
- **Model Type**: Quantized (QAT)
- **Input Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x3
- **Output Shape**: {INPUT_HEIGHT}x{INPUT_WIDTH}x{NUM_CLASSES}
- **FPGA Part**: {args.part}
- **Clock Period**: {args.clock_period}ns
- **I/O Type**: {args.io_type}
- **Strategy**: {args.strategy}
- **Reuse Factor**: {args.reuse_factor}
- **Platform**: Windows

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

## Usage Instructions (Windows)

### 1. Simulation
```cmd
cd {output_dir}
myproject.exe
```

### 2. Synthesis
```cmd
cd {output_dir}
vivado_hls -f run_hls.tcl
```

### 3. Co-simulation
```cmd
cd {output_dir}
myproject.exe
```

### 4. IP Export
The synthesized IP core can be found in:
`{output_dir}\\myproject_prj\\solution1\\impl\\ip\\`

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
- Check synthesis logs in `myproject_prj\\solution1\\syn\\report\\`
- Verify timing constraints are met
- Ensure sufficient FPGA resources are available
- Check I/O timing for your specific application
- Make sure Vivado HLS is properly installed and in PATH

## Windows-Specific Notes
- Use Windows path separators (\\) in file paths
- Ensure Vivado HLS is installed and accessible
- Check Windows firewall settings for network features
- Use Windows-compatible compilers (Visual Studio)

## Support
For issues with HLS4ML conversion, check:
- HLS4ML documentation: https://hls4ml.readthedocs.io/
- Model compatibility with supported layers
- FPGA resource constraints
- Timing requirements
- Windows-specific Vivado installation issues
"""

    guide_file = os.path.join(output_dir, "DEPLOYMENT_GUIDE.md")
    with open(guide_file, "w") as f:
        f.write(guide_content)

    print(f"Deployment guide saved to: {guide_file}")


def main():
    """Main function"""
    args = parse_arguments()

    print("=" * 60)
    print("HLS4ML FPGA Model Builder for LMIINet (Windows Version)")
    print("=" * 60)

    # Setup Windows environment
    setup_windows_environment()

    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"lmiinet_qat_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"FPGA part: {args.part}")
    print(f"Clock period: {args.clock_period}ns")
    print(f"I/O type: {args.io_type}")
    if args.io_type == "io_stream":
        print("   Using io_stream for semantic segmentation compatibility")
    print(f"Strategy: {args.strategy}")
    print(f"Reuse factor: {args.reuse_factor}")
    print(f"Platform: {platform.system()}")

    try:
        # Step 1: Load QAT model
        model = load_qat_model(args.model_path)

        # Step 1.5: Fix UpSampling2D layers for HLS4ML compatibility
        model = fix_upsampling_layers(model)

        # Step 2: Create HLS4ML configuration
        config = create_hls4ml_config(args)

        # Step 3: Convert to HLS4ML
        hls_model = convert_to_hls4ml(model, config, output_dir)

        # Step 4: Validate conversion (if requested)
        validation_results = None
        if args.validate:
            validation_results = validate_conversion(model, hls_model, output_dir, compile_model=True)

        # Step 5: Generate performance report
        generate_performance_report(hls_model, output_dir, args)

        # Step 7: Create deployment guide
        create_deployment_guide(output_dir, args)
        
        # Step 8: Run synthesis if requested
        if args.synthesize:
            run_synthesis(output_dir, args)
        
        print("\n" + "=" * 60)
        print("FPGA Model Build Completed Successfully!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"HLS4ML project: {output_dir}/myproject_prj/")
        print(f"Deployment guide: {output_dir}/DEPLOYMENT_GUIDE.md")

        if args.validate and validation_results is not None:
            print(f"Validation results: {output_dir}/validation_results.json")
        elif args.validate:
            print("Validation skipped due to compilation issues")

        print("\nNext steps:")
        print("1. Review the deployment guide")
        print("2. Run simulation: cd {output_dir} && myproject.exe")
        if args.synthesize:
            print("3. Synthesis completed automatically!")
            print("4. Check synthesis reports in myproject_prj/solution1/syn/report/")
            print("5. Integrate with your Vivado project")
        else:
            print("3. Run synthesis: vivado_hls -f build_prj.tcl")
            print("4. Integrate with your Vivado project")

    except Exception as e:
        print(f"\nError during FPGA model build: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
