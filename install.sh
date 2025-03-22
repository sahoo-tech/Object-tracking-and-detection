#!/bin/bash

# Exit on error
set -e

echo "Installing TrainIT - Advanced Object Detection and Tracking System"
echo "=============================================================="

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/models
mkdir -p data/output/videos
mkdir -p data/output/analytics
mkdir -p data/output/logs

# Download YOLOv8 weights if they don't exist
if [ ! -f "data/models/yolov8n.pt" ]; then
    echo "Downloading YOLOv8 weights..."
    wget -P data/models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA is available' if torch.cuda.is_available() else 'CUDA is not available')"

# Create default config if it doesn't exist
if [ ! -f "configs/default.yaml" ]; then
    echo "Creating default configuration..."
    cp configs/default.yaml.example configs/default.yaml
fi

echo "=============================================================="
echo "Installation complete!"
echo "To start using TrainIT:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the application: python -m src.main"
echo "==============================================================" 