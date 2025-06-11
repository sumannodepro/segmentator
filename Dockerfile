# Use NVIDIA CUDA runtime as base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python, pip, and other system packages
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Symlink python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# Copy code and model/data folders
COPY inference.py ./
COPY nnUNet_data ./nnUNet_data
COPY input ./input
COPY output ./output

# Set nnU-Net environment variables
ENV nnUNet_raw=/app/nnUNet_data/nnUNet_raw
ENV nnUNet_preprocessed=/app/nnUNet_data/nnUNet_preprocessed
ENV nnUNet_results=/app/nnUNet_data/nnUNet_results

# Default command
CMD ["python", "inference.py"]
