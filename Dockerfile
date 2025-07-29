# Use official TensorFlow image with CUDA support
FROM tensorflow/tensorflow:2.15.0-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional HLS4ML dependencies
RUN pip install hls4ml[profiling]

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Create directories for data and models
RUN mkdir -p /data /app/models

# Set environment variables for TensorFlow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Default command
CMD ["tail", "-f", "/dev/null"] 