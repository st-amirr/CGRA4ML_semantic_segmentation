# CGRA4ML Installation Guide for LMIINet

## Python Version Compatibility Issue

CGRA4ML has a strict Python version requirement: **Python 3.11.5 exactly**. If you're getting the error:
```
ERROR: Package 'deepsocflow' requires a different Python: 3.11.9 not in '==3.11.5'
```

This guide provides multiple solutions to resolve this issue.

## Solution 1: Use Conda (Recommended) ⭐

Conda allows you to create isolated environments with specific Python versions.

### Prerequisites
1. Install Anaconda or Miniconda from: https://docs.conda.io/en/latest/miniconda.html

### Installation Steps

```bash
# 1. Run the conda setup script
setup_cgra4ml_conda.bat

# 2. Activate the environment
activate_cgra4ml_conda.bat

# 3. Test installation
test_cgra4ml_conda.bat
```

### Manual Conda Setup

```bash
# Create environment with Python 3.11.5
conda create -n cgra4ml_env python=3.11.5 -y

# Activate environment
conda activate cgra4ml_env

# Install dependencies
pip install -r requirements_cgra4ml.txt

# Clone and install CGRA4ML
git clone https://github.com/KastnerRG/cgra4ml.git
cd cgra4ml
pip install -e .
cd ..
```

## Solution 2: Use pyenv (Alternative)

pyenv allows you to manage multiple Python versions.

### Installation

```bash
# Install pyenv (Windows)
# Download from: https://github.com/pyenv-win/pyenv-win

# Install Python 3.11.5
pyenv install 3.11.5

# Set local Python version
pyenv local 3.11.5

# Create virtual environment
python -m venv cgra4ml_env
cgra4ml_env\Scripts\activate.bat

# Install CGRA4ML
pip install -r requirements_cgra4ml.txt
git clone https://github.com/KastnerRG/cgra4ml.git
cd cgra4ml
pip install -e .
cd ..
```

## Solution 3: Manual Installation with Override

If you want to force installation with your current Python version:

```bash
# 1. Create virtual environment
python -m venv cgra4ml_env
cgra4ml_env\Scripts\activate.bat

# 2. Install dependencies
pip install -r requirements_cgra4ml.txt

# 3. Clone CGRA4ML
git clone https://github.com/KastnerRG/cgra4ml.git

# 4. Install with version override
cd cgra4ml
pip install -e . --ignore-installed --no-deps
cd ..
```

**Warning**: This may cause compatibility issues with CGRA4ML dependencies.

## Solution 4: Docker Container

Create a Docker container with the exact Python version:

```dockerfile
FROM python:3.11.5-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone CGRA4ML
RUN git clone https://github.com/KastnerRG/cgra4ml.git

# Install CGRA4ML
WORKDIR /app/cgra4ml
RUN pip install -e .

# Copy your project
WORKDIR /app
COPY . .

# Install project dependencies
RUN pip install -r requirements_cgra4ml.txt
```

## Verification Steps

After installation, verify everything works:

```bash
# Test CGRA4ML import
python -c "import cgra4ml; print('CGRA4ML imported successfully')"

# Test model building
python build_cgra4ml_model.py --model_path ./models/qat_model_clean_for_compatible.h5 --validate
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'cgra4ml'**
   ```bash
   # Solution: Reinstall CGRA4ML
   cd cgra4ml
   pip install -e . --force-reinstall
   cd ..
   ```

2. **Version Conflict with deepsocflow**
   ```bash
   # Solution: Use conda environment with Python 3.11.5
   conda create -n cgra4ml_env python=3.11.5 -y
   conda activate cgra4ml_env
   ```

3. **Build Dependencies Missing**
   ```bash
   # Install build tools
   pip install --upgrade pip setuptools wheel
   ```

4. **Git Clone Failed**
   ```bash
   # Check internet connection
   # Try with SSH if HTTPS fails
   git clone git@github.com:KastnerRG/cgra4ml.git
   ```

### Environment Variables

Set these environment variables for CGRA4ML:

```bash
# Windows
set CGRA4ML_BACKEND=Vivado
set CGRA4ML_PART=xc7a200tfbg676-1

# Linux/Mac
export CGRA4ML_BACKEND=Vivado
export CGRA4ML_PART=xc7a200tfbg676-1
```

## Hardware Tools Setup

CGRA4ML requires additional hardware tools:

### Verilator (Simulation)
```bash
# Windows: Download from https://veripool.org/verilator/
# Linux: sudo apt-get install verilator
# Mac: brew install verilator
```

### Vivado (Synthesis)
- Download from Xilinx website
- Add to PATH: `C:\Xilinx\Vivado\2023.2\bin`

### C++ Compiler
```bash
# Windows: Install Visual Studio Build Tools
# Linux: sudo apt-get install build-essential
# Mac: Install Xcode Command Line Tools
```

## Next Steps

After successful installation:

1. **Test the installation**: `test_cgra4ml_conda.bat`
2. **Build your model**: 
   ```bash
   python build_cgra4ml_model.py \
       --model_path ./models/qat_model_clean_for_compatible.h5 \
       --output_dir ./cgra4ml_models \
       --validate
   ```
3. **Run synthesis**:
   ```bash
   python build_cgra4ml_model.py \
       --model_path ./models/qat_model_clean_for_compatible.h5 \
       --output_dir ./cgra4ml_models \
       --synthesize
   ```

## Support

If you encounter issues:

1. Check the CGRA4ML repository: https://github.com/KastnerRG/cgra4ml
2. Verify Python version: `python --version`
3. Check environment: `conda list` or `pip list`
4. Review error logs for specific issues

## Quick Reference

| Method | Python Version | Pros | Cons |
|--------|---------------|------|------|
| Conda | 3.11.5 | Isolated, reliable | Requires conda |
| pyenv | 3.11.5 | Version management | Setup complexity |
| Manual Override | Any | Quick | Compatibility issues |
| Docker | 3.11.5 | Reproducible | Resource overhead |
