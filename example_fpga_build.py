#!/usr/bin/env python3
"""
Example Script: Building FPGA Models with HLS4ML
===============================================

This script demonstrates how to use the build_fpga_model.py script
to convert trained LMIINet models to FPGA implementations.

Examples:
1. Convert float32 model with basic settings
2. Convert QAT model with optimized settings
3. Convert with validation and synthesis
4. Convert with custom FPGA part and precision

Usage:
    python example_fpga_build.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_fpga_build(model_path, is_quantized=False, **kwargs):
    """
    Run the FPGA model builder with specified parameters
    
    Args:
        model_path: Path to the trained model
        is_quantized: Whether the model is quantized
        **kwargs: Additional arguments for build_fpga_model.py
    """
    cmd = [
        sys.executable, 'build_fpga_model.py',
        '--model_path', model_path
    ]
    
    if is_quantized:
        cmd.append('--quantized')
    
    # Add additional arguments
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Command completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def example_1_basic_float32():
    """Example 1: Convert float32 model with basic settings"""
    print("\n" + "="*60)
    print("Example 1: Basic Float32 Model Conversion")
    print("="*60)
    
    model_path = "./models/float32_model_final.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a float32 model first using:")
        print("python trainQAT.py --trainQAT=False --epochs=50")
        return False
    
    success = run_fpga_build(
        model_path=model_path,
        is_quantized=False,
        output_dir="./fpga_models/example1_basic",
        validate=True
    )
    
    if success:
        print("✅ Basic float32 conversion completed")
        print("Check output in: ./fpga_models/example1_basic/")
    
    return success

def example_2_qat_optimized():
    """Example 2: Convert QAT model with optimized settings"""
    print("\n" + "="*60)
    print("Example 2: QAT Model with Optimized Settings")
    print("="*60)
    
    model_path = "./models/qat_model_final.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a QAT model first using:")
        print("python trainQAT.py --trainQAT=True --epochs=30 --load_model=./models/float32_model_final.keras")
        return False
    
    success = run_fpga_build(
        model_path=model_path,
        is_quantized=True,
        output_dir="./fpga_models/example2_qat",
        strategy="Resource",
        reuse_factor=2,
        precision="ap_fixed<12,4>",
        validate=True
    )
    
    if success:
        print("✅ QAT model conversion completed")
        print("Check output in: ./fpga_models/example2_qat/")
    
    return success

def example_3_full_synthesis():
    """Example 3: Convert with validation and synthesis"""
    print("\n" + "="*60)
    print("Example 3: Full Conversion with Synthesis")
    print("="*60)
    
    # Try QAT model first, fallback to float32
    model_path = "./models/qat_model_final.keras"
    is_quantized = True
    
    if not os.path.exists(model_path):
        model_path = "./models/float32_model_final.keras"
        is_quantized = False
        print(f"QAT model not found, using float32 model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ No trained model found")
        print("Please train a model first")
        return False
    
    success = run_fpga_build(
        model_path=model_path,
        is_quantized=is_quantized,
        output_dir="./fpga_models/example3_full",
        validate=True,
        synthesize=True,
        strategy="Latency",
        clock_period=4.0
    )
    
    if success:
        print("✅ Full conversion with synthesis completed")
        print("Check output in: ./fpga_models/example3_full/")
    
    return success

def example_4_custom_fpga():
    """Example 4: Convert with custom FPGA part and settings"""
    print("\n" + "="*60)
    print("Example 4: Custom FPGA Configuration")
    print("="*60)
    
    # Try to find any trained model
    possible_models = [
        "./models/qat_model_final.keras",
        "./models/float32_model_final.keras",
        "./models/qat_model_best.keras",
        "./models/float32_model_best.keras"
    ]
    
    model_path = None
    is_quantized = False
    
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            is_quantized = "qat" in path
            break
    
    if model_path is None:
        print("❌ No trained model found")
        print("Please train a model first")
        return False
    
    print(f"Using model: {model_path}")
    
    success = run_fpga_build(
        model_path=model_path,
        is_quantized=is_quantized,
        output_dir="./fpga_models/example4_custom",
        part="xc7z020clg484-1",  # Zynq-7020
        clock_period=6.0,
        io_type="io_stream",
        strategy="Resource",
        reuse_factor=4,
        precision="ap_fixed<14,5>",
        validate=True
    )
    
    if success:
        print("✅ Custom FPGA conversion completed")
        print("Check output in: ./fpga_models/example4_custom/")
    
    return success

def check_prerequisites():
    """Check if required tools and models are available"""
    print("Checking prerequisites...")
    
    # Check if build_fpga_model.py exists
    if not os.path.exists("build_fpga_model.py"):
        print("❌ build_fpga_model.py not found")
        return False
    
    # Check if trainQAT.py exists
    if not os.path.exists("trainQAT.py"):
        print("❌ trainQAT.py not found")
        return False
    
    # Check for trained models
    models_dir = Path("./models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        print("Please train a model first")
        return False
    
    # Look for any trained models
    model_files = list(models_dir.glob("*_final.keras")) + list(models_dir.glob("*_best.keras"))
    
    if not model_files:
        print("❌ No trained models found")
        print("Please train a model first using:")
        print("python trainQAT.py --trainQAT=False --epochs=50")
        return False
    
    print("✅ Prerequisites check passed")
    print(f"Found {len(model_files)} trained model(s):")
    for model in model_files:
        print(f"  - {model}")
    
    return True

def main():
    """Main function to run all examples"""
    print("="*60)
    print("HLS4ML FPGA Model Builder - Examples")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Create output directory
    os.makedirs("./fpga_models", exist_ok=True)
    
    # Run examples
    examples = [
        ("Basic Float32", example_1_basic_float32),
        ("QAT Optimized", example_2_qat_optimized),
        ("Full Synthesis", example_3_full_synthesis),
        ("Custom FPGA", example_4_custom_fpga)
    ]
    
    results = []
    
    for name, example_func in examples:
        print(f"\n{'='*20} Running {name} Example {'='*20}")
        try:
            success = example_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Error in {name} example: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("Example Results Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {successful}/{total} examples completed successfully")
    
    if successful > 0:
        print("\n🎉 FPGA model building examples completed!")
        print("Check the ./fpga_models/ directory for generated projects")
        print("\nNext steps:")
        print("1. Review the generated deployment guides")
        print("2. Run simulations in the generated directories")
        print("3. Integrate with your Vivado project")
    else:
        print("\n❌ No examples completed successfully")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
