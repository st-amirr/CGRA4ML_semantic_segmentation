@echo off
REM HLS4ML Windows Environment Setup Script
REM =======================================

echo ================================================
echo    HLS4ML Windows Environment Setup
echo ================================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist hls4ml_env (
    echo Virtual environment already exists
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "%recreate%"=="y" (
        echo Removing existing environment...
        rmdir /s /q hls4ml_env
    ) else (
        echo Using existing environment
        goto :install_deps
    )
)

python -m venv hls4ml_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully!

:install_deps
REM Activate virtual environment and install dependencies
echo.
echo Activating virtual environment and installing dependencies...
call hls4ml_env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements_windows.txt...
pip install -r requirements_windows.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================
echo    Setup completed successfully!
echo ================================================
echo.
echo To activate the environment in the future:
echo   hls4ml_env\Scripts\activate.bat
echo.
echo To run the FPGA model builder:
echo   python build_fpga_model_windows.py --model_path ./models/your_model.h5 --validate
echo.
echo To deactivate the environment:
echo   deactivate
echo.

pause
