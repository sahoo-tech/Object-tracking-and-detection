# API Documentation

## Classes

### RealTimeObjectTracker

Main class for object detection and tracking.

#### Constructor Parameters
- `mode` (TrackingMode): Operation mode (NORMAL, SAVE_VIDEO, ANALYTICS, ADVANCED)

#### Methods

##### `__init__(mode: TrackingMode = TrackingMode.ADVANCED)`
Initializes the tracker with specified mode.

##### `run()`
Starts the tracking system.

##### `process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Detections]`
Processes a single frame.

##### `_initialize_camera()`
Initializes and configures the camera.

##### `_process_frames()`
Background thread for frame processing.

##### `_update_trajectories(detections: Detections, frame_time: float)`
Updates object trajectories.

##### `_update_heatmap(detections: Detections)`
Updates the heatmap data.

##### `_update_interactions(detections: Detections)`
Updates object interaction data.

### ObjectTrajectory

Class for storing and analyzing object trajectories.

#### Constructor Parameters
- `positions`: List of position tuples
- `velocities`: List of velocity tuples
- `timestamps`: List of timestamps
- `class_id`: Object class ID
- `track_id`: Tracking ID
- `confidence_history`: List of confidence values

#### Methods

##### `predict_next_position(time_ahead: float = 0.1) -> Tuple[float, float]`
Predicts future position of object.

##### `update_motion_pattern()`
Updates the motion pattern classification.

### AdvancedVisualizer

Class for visualization features.

#### Constructor Parameters
- `frame_size`: Tuple of frame dimensions

#### Methods

##### `draw_trajectory(frame: np.ndarray, trajectory: ObjectTrajectory) -> np.ndarray`
Draws object trajectory.

##### `draw_heatmap(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray`
Draws heatmap overlay.

##### `draw_interaction_lines(frame: np.ndarray, trajectories: Dict, interaction_matrix: Dict) -> np.ndarray`
Draws interaction lines between objects.

##### `annotate_frame(frame: np.ndarray, detections: Detections, model_names: List[str] = None) -> np.ndarray`
Annotates frame with detection boxes and labels.

### TrackingStats

Class for storing tracking statistics.

#### Properties
- `total_objects`: Total objects detected
- `unique_objects`: Dictionary of unique object counts
- `frame_count`: Total frames processed
- `fps_history`: List of FPS values
- `trajectories`: Dictionary of object trajectories
- `heatmap_data`: Numpy array of heatmap
- `interaction_matrix`: Dictionary of object interactions
- `object_velocities`: Dictionary of object velocities

#### Methods

##### `update_interaction_history(frame_time: float)`
Updates interaction history.

##### `get_interaction_statistics() -> Dict`
Returns interaction statistics.

## Enums

### TrackingMode
- `NORMAL`: Basic tracking mode
- `SAVE_VIDEO`: Records video output
- `ANALYTICS`: Enables analytics collection
- `ADVANCED`: Enables all features

## Constants

### Camera Settings
- Frame Width: 1280
- Frame Height: 720
- FPS: 30
- Confidence Threshold: 0.5

### Tracking Parameters
- Track High Threshold: 0.5
- Track Low Threshold: 0.1
- New Track Threshold: 0.6
- Track Buffer: 30
- Match Threshold: 0.8

### Visualization Parameters
- Trajectory Length: 30
- Heatmap Alpha: 0.3
- Max Text Width: 200
- Trajectory Alpha: 0.7
- Interaction Alpha: 0.5
- Prediction Alpha: 0.8 