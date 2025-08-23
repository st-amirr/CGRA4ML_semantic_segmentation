# HLS4ML Windows Setup Guide

## Quick Start

1. **Run the setup script:**
   ```cmd
   setup_windows.bat
   ```

2. **Activate the environment:**
   ```cmd
   hls4ml_env\Scripts\activate.bat
   ```

3. **Run the FPGA model builder:**
   ```cmd
   python build_fpga_model_windows.py --model_path ./models/your_model.h5 --validate
   python build_fpga_model_windows.py --model_path ./models/qat_model_clean_for_compatible.h5 --validate
   ```

## Files Created

- **`build_fpga_model_windows.py`** - Windows-compatible FPGA model builder
- **`requirements_windows.txt`** - Windows dependencies
- **`setup_windows.bat`** - Automated setup script
- **`hls4ml_env/`** - Virtual environment (created by setup)

## Prerequisites

- Python 3.8+ installed
- Vivado HLS installed (for synthesis)
- Windows 10/11

## Usage Examples

### Basic conversion:
```cmd
python build_fpga_model_windows.py --model_path ./models/qat_model.h5
```

### With validation:
```cmd
python build_fpga_model_windows.py --model_path ./models/qat_model.h5 --validate
```

### With synthesis:
```cmd
python build_fpga_model_windows.py --model_path ./models/qat_model.h5 --validate --synthesize
```

## Troubleshooting

- **Python not found**: Install Python 3.8+ and add to PATH
- **Vivado not found**: Install Vivado HLS or set VIVADO_HLS_PATH
- **Dependencies failed**: Check internet connection and try again
