# Advanced Real-Time Object Detection and Tracking

A production-ready real-time object detection and tracking system built with Python, YOLOv8, and OpenCV. The system provides advanced motion analysis, object interaction tracking, and comprehensive visualization features.

## Project Structure

```
trainit/
│
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── detector.py    # YOLOv8 detector
│   │   ├── tracker.py     # ByteTrack tracker
│   │   └── camera.py      # Camera handling
│   │
│   ├── analysis/          # Analysis modules
│   │   ├── motion.py      # Motion analysis
│   │   ├── interaction.py # Object interaction
│   │   └── patterns.py    # Pattern recognition
│   │
│   └── visualization/     # Visualization modules
│       ├── display.py     # Main display handling
│       ├── heatmap.py     # Heatmap generation
│       └── annotator.py   # Frame annotation
│
├── configs/               # Configuration files
│   ├── default.yaml      # Default settings
│   └── advanced.yaml     # Advanced settings
│
├── data/                 # Data directory
│   ├── models/          # Model weights
│   └── output/          # Output files
│
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
│
├── utils/              # Utility functions
│   ├── logger.py      # Logging setup
│   └── helpers.py     # Helper functions
│
├── scripts/            # Scripts
│   ├── install.sh     # Installation script
│   └── download_weights.sh  # Model download
│
├── docs/              # Documentation
│   ├── api/          # API documentation
│   ├── guides/       # User guides
│   └── examples/     # Code examples
│
├── requirements.txt   # Project dependencies
├── setup.py          # Package setup
└── README.md         # Project documentation
```

## Project Setup

### Environment Setup
1. **System Requirements**:
   - Python 3.8 or higher
   - CUDA-capable GPU (recommended)
   - Webcam with 720p or higher resolution
   - 8GB RAM minimum (16GB recommended)

2. **Directory Structure**:
   - `src/`: Core implementation files
   - `data/`: Model weights and output storage
   - `configs/`: Configuration files
   - `tests/`: Test suites
   - `utils/`: Helper utilities
   - `scripts/`: Installation and setup scripts

3. **Development Tools**:
   - VS Code/PyCharm for development
   - Git for version control
   - pytest for testing
   - black for code formatting
   - mypy for type checking

## Datasets Used

### Training Data
1. **COCO Dataset**:
   - 80+ object classes
   - Over 200K labeled images
   - Instance segmentation
   - Used for YOLOv8 training

2. **Model Weights**:
   - YOLOv8n.pt (default)
   - YOLOv8s.pt (small)
   - YOLOv8m.pt (medium)
   - YOLOv8l.pt (large)
   - YOLOv8x.pt (xlarge)

3. **Runtime Data**:
   - Real-time webcam feed
   - Video file input support
   - Analytics data in JSON format
   - Performance metrics logs

## Tools and Technologies

### Core Technologies
1. **Computer Vision**:
   - OpenCV 4.8.0+
   - YOLOv8 by Ultralytics
   - ByteTrack for object tracking
   - NumPy for numerical operations

2. **Deep Learning**:
   - PyTorch 2.0+
   - CUDA for GPU acceleration
   - TorchVision for image processing
   - Supervision for detection utils

3. **Data Processing**:
   - Pandas for analytics
   - Matplotlib for plotting
   - SciPy for scientific computing
   - PyYAML for configuration

4. **Development Tools**:
   - pytest for testing
   - black for formatting
   - mypy for type checking
   - logging for debug info

## Execution Instructions

### Basic Usage
1. **Starting the System**:
   ```bash
   # Activate environment
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   # Run application
   python -m src.main
   ```

2. **Operation Modes**:
   ```bash
   # Normal mode
   python -m src.main --mode normal

   # Video recording
   python -m src.main --mode save_video

   # Analytics collection
   python -m src.main --mode analytics

   # Advanced features
   python -m src.main --mode advanced
   ```

3. **Configuration Options**:
   ```bash
   # Use custom config
   python -m src.main --config configs/custom.yaml

   # Override settings
   python -m src.main --confidence 0.6 --device cuda
   ```

### Advanced Usage
1. **Custom Analysis**:
   ```bash
   # Enable specific features
   python -m src.main --enable-heatmap --enable-tracking

   # Set analysis parameters
   python -m src.main --trajectory-length 50 --prediction-horizon 1.0
   ```

2. **Output Options**:
   ```bash
   # Specify output directory
   python -m src.main --output-dir /path/to/output

   # Set video format
   python -m src.main --video-format mp4 --video-fps 30
   ```

3. **Performance Tuning**:
   ```bash
   # Adjust thread count
   python -m src.main --num-threads 4

   # Set GPU memory fraction
   python -m src.main --gpu-memory-fraction 0.8
   ```

## Features

### Core Functionality
- Real-time object detection using YOLOv8
- Advanced object tracking with ByteTrack
- Motion prediction and pattern analysis
- Object interaction detection and visualization
- Dynamic heatmap generation
- Multi-threaded processing for optimal performance

### Advanced Features
- 8-directional motion prediction
- Real-time velocity and acceleration analysis
- Interaction matrix and proximity detection
- Gradient-colored trajectory visualization
- Performance monitoring and analytics
- Multiple operation modes (Normal, Save Video, Analytics, Advanced)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/sahoo-tech/Object-tracking-and-detection.git
cd Object-tracking-and-detection
```

2. Run the installation script:
```bash
# Linux/Mac
./scripts/install.sh

# Windows
scripts\install.bat
```

3. Start the application:
```bash
python -m src.main
```

## Configuration

The system can be configured through YAML files in the `configs/` directory:

```yaml
# configs/default.yaml
detector:
  model: yolov8n
  confidence: 0.5
  device: cuda

tracker:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

visualization:
  trajectory_length: 30
  heatmap_alpha: 0.3
```

## Development

### Setting up the development environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings in Google format
- Keep functions focused and small

## Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack implementation
- OpenCV community
- PyTorch framework

## Contact

Your Name - [@SayantanSahoo]
Project Link: (https://github.com/sahoo-tech/Object-tracking-and-detection.git)
