@echo off
REM CGRA4ML Setup Script for LMIINet Semantic Segmentation
REM ======================================================

echo ========================================
echo CGRA4ML Setup for LMIINet
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv cgra4ml_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call cgra4ml_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements_cgra4ml.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

REM Clone CGRA4ML repository
echo.
echo Cloning CGRA4ML repository...
if not exist cgra4ml (
    git clone https://github.com/KastnerRG/cgra4ml.git
    if errorlevel 1 (
        echo ERROR: Failed to clone CGRA4ML repository
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
) else (
    echo CGRA4ML repository already exists, updating...
    cd cgra4ml
    git pull
    cd ..
)

REM Check Python version compatibility
echo.
echo Checking Python version compatibility...
python -c "import sys; version=sys.version_info; print(f'Python {version.major}.{version.minor}.{version.micro}')"
python -c "import sys; version=sys.version_info; exit(0 if version.major==3 and version.minor==11 and version.micro==5 else 1)"
if errorlevel 1 (
    echo WARNING: CGRA4ML requires Python 3.11.5 exactly
    echo Current Python version may cause compatibility issues
    echo.
    echo Attempting to install with version override...
)

REM Fix CGRA4ML requirements for Python 3.11.9 compatibility
echo.
echo Fixing CGRA4ML requirements for Python 3.11.9...

REM Create backup of pyproject.toml
if exist cgra4ml\pyproject.toml (
    if not exist cgra4ml\pyproject.toml.backup (
        copy cgra4ml\pyproject.toml cgra4ml\pyproject.toml.backup
        echo Created backup: cgra4ml\pyproject.toml.backup
    )
)

REM Fix pyproject.toml using PowerShell
echo Fixing pyproject.toml file...
powershell -Command "if (Test-Path 'cgra4ml\pyproject.toml') { $content = Get-Content 'cgra4ml\pyproject.toml' -Raw; $content = $content -replace 'python\s*=\s*\"3\.11\.5\"', 'python = \">=3.11.0\"'; $content = $content -replace 'requires-python\s*=\s*\"==3\.11\.5\"', 'requires-python = \">=3.11.0\"'; $content = $content -replace 'requires-python\s*=\s*\"3\.11\.5\"', 'requires-python = \">=3.11.0\"'; $content = $content -replace '\"3\.11\.5\"', '\"3.11.0\"'; $content = $content -replace 'deepsocflow\s*=\s*\"[^\"]*\"', 'deepsocflow = \"*\"'; Set-Content 'cgra4ml\pyproject.toml' $content; Write-Host 'Fixed pyproject.toml successfully' } else { Write-Host 'pyproject.toml not found' }"

if errorlevel 1 (
    echo.
    echo WARNING: Could not fix pyproject.toml automatically
    echo Will attempt installation with version override...
)

REM Install CGRA4ML after fixing requirements
echo.
echo Installing CGRA4ML with fixed requirements...
cd cgra4ml

REM Try multiple installation methods
echo Trying standard installation...
pip install -e . --no-deps --ignore-installed

if errorlevel 1 (
    echo Standard installation failed, trying alternative methods...
    
    echo Trying with setup.py...
    if exist setup.py (
        python setup.py develop
    )
    
    if errorlevel 1 (
        echo Setup.py failed, trying force install...
        pip install -e . --force-reinstall --no-deps --ignore-installed
        
        if errorlevel 1 (
            echo.
            echo ERROR: All installation methods failed!
            echo.
            echo RECOMMENDED SOLUTIONS:
            echo 1. Use Python 3.11.5 exactly (install from python.org)
            echo 2. Use conda: conda create -n cgra4ml_env python=3.11.5
            echo 3. Try manual installation: pip install -e . --ignore-installed --no-deps
            echo.
            pause
            exit /b 1
        )
    )
)

cd ..

REM Check for hardware tools
echo.
echo Checking for hardware tools...

REM Check for Verilator
verilator --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Verilator not found in PATH
    echo Please install Verilator 5.014+ for simulation
) else (
    echo Verilator found:
    verilator --version
)

REM Check for Vivado
vivado -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Vivado not found in PATH
    echo Please install Xilinx Vivado for synthesis
) else (
    echo Vivado found:
    vivado -version
)

REM Check for C++ compiler
g++ --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: C++ compiler (g++) not found in PATH
    echo Please install a C++ compiler for compilation
) else (
    echo C++ compiler found:
    g++ --version
)

REM Create test script
echo.
echo Creating test script...
echo @echo off > test_cgra4ml.bat
echo echo Testing CGRA4ML installation... >> test_cgra4ml.bat
echo call cgra4ml_env\Scripts\activate.bat >> test_cgra4ml.bat
echo python -c "import cgra4ml; print('CGRA4ML imported successfully')" >> test_cgra4ml.bat
echo pause >> test_cgra4ml.bat

echo.
echo ========================================
echo CGRA4ML Setup Complete!
echo ========================================
echo.
echo To activate the environment, run:
echo   cgra4ml_env\Scripts\activate.bat
echo.
echo To test the installation, run:
echo   test_cgra4ml.bat
echo.
echo To build your LMIINet model with CGRA4ML:
echo   python build_cgra4ml_model.py --model_path ./models/qat_model_clean_for_compatible.h5 --validate
echo.
echo ========================================
pause
