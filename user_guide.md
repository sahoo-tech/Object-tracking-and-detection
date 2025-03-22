# User Guide

## Getting Started

### System Requirements

Before running the system, ensure you have:
1. Python 3.8 or higher installed
2. A webcam (built-in or USB)
3. NVIDIA GPU with CUDA support (recommended)
4. Sufficient RAM (8GB minimum, 16GB recommended)

### Installation

1. Set up your environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

2. Download YOLOv8 weights:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Running the System

### Basic Usage

1. Start the system in advanced mode:
```bash
python real_time_tracking.py
```

2. The system will:
   - Initialize the camera
   - Load the YOLO model
   - Start real-time detection and tracking
   - Display the video feed with annotations

3. To exit:
   - Press 'q' key
   - Or close the window

### Operation Modes

#### 1. Normal Mode
- Basic detection and tracking
- Minimal visualization
```python
tracker = RealTimeObjectTracker(mode=TrackingMode.NORMAL)
```

#### 2. Save Video Mode
- Records session to MP4 file
- Includes all visualizations
```python
tracker = RealTimeObjectTracker(mode=TrackingMode.SAVE_VIDEO)
```

#### 3. Analytics Mode
- Collects detailed statistics
- Saves analytics to JSON
```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ANALYTICS)
```

#### 4. Advanced Mode
- All features enabled
- Full visualization
```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)
```

## Understanding the Display

### Main Display Elements

1. **Detection Boxes**
   - Colored boxes around detected objects
   - Labels showing class name and confidence
   - Unique tracking ID

2. **Trajectories**
   - Gradient-colored lines showing object paths
   - Arrow indicators for movement direction
   - Predicted position markers

3. **Heatmap**
   - Color overlay showing object presence
   - Brighter areas indicate more activity
   - Fades over time

4. **Statistics Display**
   - FPS counter
   - Object count
   - Average velocity
   - Predicted directions

5. **Interaction Lines**
   - Lines between interacting objects
   - Interaction count display
   - Color-coded by interaction type

### Motion Indicators

The system uses 8 cardinal directions:
- → (Right)
- ↗ (Up-Right)
- ↑ (Up)
- ↖ (Up-Left)
- ← (Left)
- ↙ (Down-Left)
- ↓ (Down)
- ↘ (Down-Right)

## Analytics

### Real-time Analytics

The system displays:
1. Current FPS
2. Number of detected objects
3. Average object velocity
4. Motion patterns
5. Interaction statistics

### Saved Analytics

In analytics mode, the system saves:
1. Session duration
2. Total frames processed
3. Average FPS
4. Object counts by class
5. Interaction history
6. Performance metrics

## Troubleshooting

### Common Issues

1. **Black Screen**
   - Check camera connection
   - Verify camera permissions
   - Restart the application

2. **Low FPS**
   - Check GPU usage
   - Close other applications
   - Consider using a smaller YOLO model

3. **Detection Issues**
   - Adjust lighting conditions
   - Check confidence threshold
   - Ensure camera is stable

4. **Camera Errors**
   - Check USB connection
   - Update camera drivers
   - Verify resolution support

### Performance Optimization

1. **GPU Optimization**
   - Use CUDA-capable GPU
   - Keep drivers updated
   - Monitor GPU temperature

2. **CPU Usage**
   - Close unnecessary applications
   - Monitor system resources
   - Adjust process priority

3. **Memory Management**
   - Monitor RAM usage
   - Clear system cache
   - Restart for long sessions

## Best Practices

1. **Camera Setup**
   - Good lighting conditions
   - Stable mounting
   - Clear field of view

2. **System Configuration**
   - Regular driver updates
   - Clean system state
   - Proper cooling

3. **Usage Guidelines**
   - Regular system checks
   - Monitor performance
   - Back up analytics data

## Advanced Features

### Custom Configuration

You can modify:
1. Detection confidence threshold
2. Tracking parameters
3. Visualization settings
4. Analytics options

### Data Export

Analytics mode provides:
1. JSON format data
2. Performance logs
3. Session statistics
4. Interaction data

### Video Recording

Save video mode features:
1. High-quality MP4 output
2. Timestamped filenames
3. All visualizations included
4. Configurable quality 