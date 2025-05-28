# Base image: PyTorch with CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    ninja-build \
    wget \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install FlashAttention (precompiled)
RUN pip install flash-attn==2.3.2 --no-build-isolation

# Clone external projects (if needed)
# Optional: You can mount these as volumes instead
RUN git clone https://github.com/HazyResearch/flash-attention.git /app/flash-attention || true
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui || true
# Replace with your own Wan2.1 repo if private
RUN git clone https://github.com/yourname/Wan2.1.git /app/wan || true

# Copy local code (gradio_app.py, start.sh, etc.)
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose Gradio port
EXPOSE 7860

# Default startup command
ENTRYPOINT ["./start.sh"]
