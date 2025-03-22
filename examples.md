# Usage Examples

## Basic Examples

### 1. Simple Object Detection and Tracking

```python
from real_time_tracking import RealTimeObjectTracker, TrackingMode

# Initialize in normal mode
tracker = RealTimeObjectTracker(mode=TrackingMode.NORMAL)

# Start tracking
tracker.run()
```

### 2. Recording Video Output

```python
# Initialize in video recording mode
tracker = RealTimeObjectTracker(mode=TrackingMode.SAVE_VIDEO)

# Start tracking and recording
tracker.run()
# Video will be saved as output_YYYYMMDD_HHMMSS.mp4
```

### 3. Collecting Analytics

```python
# Initialize in analytics mode
tracker = RealTimeObjectTracker(mode=TrackingMode.ANALYTICS)

# Start tracking and collecting data
tracker.run()
# Analytics will be saved as analytics_YYYYMMDD_HHMMSS.json
```

### 4. Advanced Mode with All Features

```python
# Initialize in advanced mode
tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)

# Start tracking with all features enabled
tracker.run()
```

## Advanced Examples

### 1. Custom Configuration

```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)

# Modify tracking parameters
tracker.tracker_config.update({
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.2,
    "new_track_thresh": 0.7,
    "track_buffer": 40,
    "match_thresh": 0.9
})

# Adjust visualization settings
tracker.visualizer.trajectory_length = 40
tracker.visualizer.heatmap_alpha = 0.4
tracker.visualizer.trajectory_alpha = 0.8

tracker.run()
```

### 2. Processing Single Frames

```python
import cv2
import numpy as np

tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)

# Read frame from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # Process single frame
    processed_frame, detections = tracker.process_frame(frame)
    
    # Display results
    cv2.imshow("Processed Frame", processed_frame)
    cv2.waitKey(1)

cap.release()
```

### 3. Accessing Analytics Data

```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ANALYTICS)

# Run for some time
tracker.run()

# Access statistics
stats = tracker.stats

# Print summary
print(f"Total objects detected: {stats.total_objects}")
print(f"Unique objects: {stats.unique_objects}")
print(f"Average FPS: {np.mean(stats.fps_history)}")

# Get interaction statistics
interaction_stats = stats.get_interaction_statistics()
print("Most frequent interactions:", interaction_stats["most_frequent"])
```

### 4. Custom Visualization

```python
import cv2
import numpy as np

tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)

def custom_visualization(frame, detections, trajectories):
    # Draw original detections
    frame = tracker.visualizer.annotate_frame(frame, detections)
    
    # Add custom overlay
    for trajectory in trajectories.values():
        if len(trajectory.positions) > 1:
            # Draw custom path
            points = np.array(trajectory.positions, dtype=np.int32)
            cv2.polylines(frame, [points], False, (0, 255, 0), 2)
            
            # Add custom labels
            last_pos = trajectory.positions[-1]
            cv2.putText(frame,
                       f"Speed: {trajectory.velocities[-1][0]:.1f}",
                       (int(last_pos[0]), int(last_pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
    
    return frame

# Use in main loop
while True:
    ret, frame = tracker.cap.read()
    if not ret:
        break
        
    processed_frame, detections = tracker.process_frame(frame)
    
    # Apply custom visualization
    final_frame = custom_visualization(
        processed_frame,
        detections,
        tracker.stats.trajectories
    )
    
    cv2.imshow("Custom Visualization", final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 5. Saving Custom Analytics

```python
import json
from datetime import datetime

tracker = RealTimeObjectTracker(mode=TrackingMode.ANALYTICS)

# Custom analytics collector
class CustomAnalytics:
    def __init__(self):
        self.data = []
    
    def update(self, frame, detections, stats):
        frame_data = {
            "timestamp": datetime.now().isoformat(),
            "objects": len(detections),
            "fps": stats.fps_history[-1] if stats.fps_history else 0,
            "trajectories": len(stats.trajectories),
            "interactions": sum(stats.interaction_matrix.values())
        }
        self.data.append(frame_data)
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)

# Use in main loop
custom_analytics = CustomAnalytics()

while True:
    ret, frame = tracker.cap.read()
    if not ret:
        break
        
    processed_frame, detections = tracker.process_frame(frame)
    
    # Update custom analytics
    custom_analytics.update(frame, detections, tracker.stats)
    
    cv2.imshow("Tracking", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save custom analytics
custom_analytics.save("custom_analytics.json")
```

## Use Case Examples

### 1. People Counting System

```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ANALYTICS)

class PeopleCounter:
    def __init__(self):
        self.total_count = 0
        self.current_count = 0
        self.entry_line_y = 360  # middle of 720p frame
    
    def update(self, trajectories):
        for track_id, trajectory in trajectories.items():
            if len(trajectory.positions) < 2:
                continue
                
            prev_pos = trajectory.positions[-2][1]  # y coordinate
            curr_pos = trajectory.positions[-1][1]  # y coordinate
            
            # Count crossing line from top to bottom
            if prev_pos < self.entry_line_y < curr_pos:
                self.total_count += 1

counter = PeopleCounter()

while True:
    ret, frame = tracker.cap.read()
    if not ret:
        break
        
    processed_frame, detections = tracker.process_frame(frame)
    
    # Update counter
    counter.update(tracker.stats.trajectories)
    
    # Draw counting line
    cv2.line(processed_frame, (0, counter.entry_line_y),
             (1280, counter.entry_line_y), (0, 255, 0), 2)
    
    # Display counts
    cv2.putText(processed_frame,
                f"Total Count: {counter.total_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    
    cv2.imshow("People Counting", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 2. Motion Pattern Analysis

```python
tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)

class MotionAnalyzer:
    def __init__(self):
        self.patterns = {
            "stationary": 0,
            "constant_velocity": 0,
            "accelerating": 0,
            "variable_motion": 0
        }
    
    def update(self, trajectories):
        self.patterns = {k: 0 for k in self.patterns}
        
        for trajectory in trajectories.values():
            if trajectory.motion_pattern in self.patterns:
                self.patterns[trajectory.motion_pattern] += 1

analyzer = MotionAnalyzer()

while True:
    ret, frame = tracker.cap.read()
    if not ret:
        break
        
    processed_frame, detections = tracker.process_frame(frame)
    
    # Update analyzer
    analyzer.update(tracker.stats.trajectories)
    
    # Display pattern statistics
    y_pos = 30
    for pattern, count in analyzer.patterns.items():
        cv2.putText(processed_frame,
                   f"{pattern}: {count}",
                   (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)
        y_pos += 30
    
    cv2.imshow("Motion Analysis", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

These examples demonstrate various ways to use and extend the tracking system for different applications. Each example can be modified and combined to create custom solutions for specific use cases. 