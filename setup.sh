#!/bin/bash

echo "🚀 HLS4ML Cityscapes Training Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU support may not work."
    echo "   Install nvidia-docker2 for GPU support."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p cityscapes

# Check if Cityscapes dataset exists
if [ ! -d "cityscapes/leftImg8bit" ] || [ ! -d "cityscapes/gtFine" ]; then
    echo "⚠️  Cityscapes dataset not found in ./cityscapes/"
    echo "   Please download and extract the Cityscapes dataset to ./cityscapes/"
    echo "   Expected structure:"
    echo "   ./cityscapes/"
    echo "   ├── leftImg8bit/"
    echo "   │   ├── train/"
    echo "   │   └── val/"
    echo "   └── gtFine/"
    echo "       ├── train/"
    echo "       └── val/"
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Place your Cityscapes dataset in ./cityscapes/"
    echo "2. Run: docker-compose up -d"
    echo "3. Enter container: docker-compose exec hls4ml-training bash"
    echo "4. Preprocess data: python pre.py"
    echo "5. Start training: python train.py"
    echo ""
    echo "📊 Monitor training with TensorBoard:"
    echo "   docker-compose exec hls4ml-training tensorboard --logdir=/app/models --host=0.0.0.0 --port=6006"
    echo "   Then visit: http://localhost:6006"
else
    echo "❌ Docker build failed!"
    exit 1
fi 