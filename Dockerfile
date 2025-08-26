FROM tensorflow/tensorflow:2.19.0-gpu

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

COPY requirements.txt .
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Create directories for data and models
RUN mkdir -p /data /app/models

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Default command
CMD ["tail", "-f", "/dev/null"]