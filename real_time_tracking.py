import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import List, Tuple, Dict, Optional, Set
import time
import json
import os
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import colorsys
from threading import Thread, Lock
import queue
import torch.nn.functional as F
from supervision.detection.core import Detections

# Configure logging with advanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('tracking.log'),
        logging.StreamHandler()
    ]
)

class TrackingMode(Enum):
    NORMAL = "normal"
    SAVE_VIDEO = "save_video"
    ANALYTICS = "analytics"
    ADVANCED = "advanced"  # New mode with all advanced features

@dataclass
class ObjectTrajectory:
    positions: List[Tuple[float, float]]
    velocities: List[Tuple[float, float]]
    timestamps: List[float]
    class_id: int
    track_id: int
    confidence_history: List[float]
    accelerations: List[Tuple[float, float]] = None
    motion_pattern: str = "unknown"
    
    def __post_init__(self):
        self.accelerations = []
        self.motion_pattern = "unknown"
    
    def predict_next_position(self, time_ahead: float = 0.1) -> Tuple[float, float]:
        if len(self.positions) < 2:
            return self.positions[-1]
        
        # Use Kalman filter-like prediction with acceleration
        last_pos = np.array(self.positions[-1])
        last_vel = np.array(self.velocities[-1])
        last_acc = np.array(self.accelerations[-1]) if self.accelerations else np.array([0.0, 0.0])
        
        # Predict position using constant acceleration model
        predicted_pos = last_pos + last_vel * time_ahead + 0.5 * last_acc * time_ahead**2
        return tuple(predicted_pos)
    
    def update_motion_pattern(self):
        if len(self.positions) < 3:
            return
        
        # Calculate average velocity and acceleration
        avg_velocity = np.mean(self.velocities[-5:], axis=0)
        avg_acceleration = np.mean(self.accelerations[-5:], axis=0) if self.accelerations else np.array([0.0, 0.0])
        
        # Determine motion pattern
        speed = np.linalg.norm(avg_velocity)
        acc_magnitude = np.linalg.norm(avg_acceleration)
        
        if speed < 1.0:
            self.motion_pattern = "stationary"
        elif acc_magnitude < 0.1:
            self.motion_pattern = "constant_velocity"
        elif acc_magnitude > 2.0:
            self.motion_pattern = "accelerating"
        else:
            self.motion_pattern = "variable_motion"

@dataclass
class TrackingStats:
    total_objects: int = 0
    unique_objects: Dict[str, int] = None
    frame_count: int = 0
    start_time: float = 0
    fps_history: List[float] = None
    trajectories: Dict[int, ObjectTrajectory] = None
    heatmap_data: np.ndarray = None
    interaction_matrix: Dict[Tuple[str, str], int] = None
    object_velocities: Dict[int, List[float]] = None
    interaction_history: List[Dict] = None
    proximity_threshold: float = 100.0  # pixels
    interaction_duration: Dict[Tuple[str, str], float] = None
    
    def __post_init__(self):
        self.unique_objects = {}
        self.fps_history = []
        self.start_time = time.time()
        self.trajectories = {}
        self.heatmap_data = np.zeros((720, 1280), dtype=np.float32)
        self.interaction_matrix = defaultdict(int)
        self.object_velocities = defaultdict(list)
        self.interaction_history = []
        self.interaction_duration = defaultdict(float)

    def update_interaction_history(self, frame_time: float):
        current_interactions = []
        for (class1, class2), count in self.interaction_matrix.items():
            if count > 0:
                interaction_data = {
                    "timestamp": time.time(),
                    "classes": (class1, class2),
                    "count": count,
                    "duration": self.interaction_duration[(class1, class2)]
                }
                current_interactions.append(interaction_data)
                self.interaction_duration[(class1, class2)] += frame_time
        
        if current_interactions:
            self.interaction_history.append(current_interactions)
            
            # Keep only last 1000 interactions
            if len(self.interaction_history) > 1000:
                self.interaction_history.pop(0)
    
    def get_interaction_statistics(self) -> Dict:
        stats = {
            "total_interactions": sum(self.interaction_matrix.values()),
            "unique_interactions": len(self.interaction_matrix),
            "most_frequent": [],
            "longest_duration": [],
            "recent_interactions": self.interaction_history[-10:] if self.interaction_history else []
        }
        
        # Get most frequent interactions
        sorted_by_count = sorted(
            self.interaction_matrix.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        stats["most_frequent"] = [
            {"classes": classes, "count": count}
            for classes, count in sorted_by_count
        ]
        
        # Get longest duration interactions
        sorted_by_duration = sorted(
            self.interaction_duration.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        stats["longest_duration"] = [
            {"classes": classes, "duration": duration}
            for classes, duration in sorted_by_duration
        ]
        
        return stats

class AdvancedVisualizer:
    def __init__(self, frame_size: Tuple[int, int]):
        self.frame_size = frame_size
        self.color_map = self._generate_color_map()
        self.trajectory_length = 30
        self.heatmap_alpha = 0.3
        self.max_text_width = 200
        self.trajectory_alpha = 0.7
        self.interaction_alpha = 0.5
        self.prediction_alpha = 0.8
        
        # Initialize box annotator with only supported parameters
        try:
            self.box_annotator = sv.BoxAnnotator(
                thickness=2,
                color=(0, 255, 0)  # Default to green
            )
        except Exception as e:
            logging.error(f"Failed to initialize BoxAnnotator: {str(e)}")
            raise
    
    def _generate_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        colors = {}
        for i in range(80):  # YOLO classes
            hue = i / 80
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors[i] = tuple(int(x * 255) for x in rgb)
        return colors
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: ObjectTrajectory) -> np.ndarray:
        try:
            if len(trajectory.positions) < 2:
                return frame
                
            # Draw trajectory line with gradient and motion pattern
            positions = np.array(trajectory.positions[-self.trajectory_length:])
            for i in range(len(positions) - 1):
                alpha = (i + 1) / len(positions) * self.trajectory_alpha
                color = self.color_map[trajectory.class_id]
                color_with_alpha = (*color, int(255 * alpha))
                
                try:
                    pt1 = tuple(map(int, positions[i]))
                    pt2 = tuple(map(int, positions[i + 1]))
                    cv2.line(frame, pt1, pt2, color_with_alpha[:3], thickness=max(1, int(2 * alpha)))
                except (ValueError, OverflowError) as e:
                    logging.warning(f"Invalid trajectory point: {str(e)}")
                    continue
            
            # Draw predicted position with uncertainty circle and motion pattern
            try:
                pred_pos = trajectory.predict_next_position()
                center = tuple(map(int, pred_pos))
                
                # Draw uncertainty circle based on motion pattern
                radius = int(10 * (1 - np.mean(trajectory.confidence_history[-5:])))
                if trajectory.motion_pattern == "accelerating":
                    color = (0, 0, 255)  # Red for accelerating
                elif trajectory.motion_pattern == "constant_velocity":
                    color = (0, 255, 0)  # Green for constant velocity
                elif trajectory.motion_pattern == "stationary":
                    color = (255, 255, 0)  # Yellow for stationary
                else:
                    color = (0, 255, 255)  # Default color
                
                cv2.circle(frame, center, radius, color, 1)
                cv2.circle(frame, center, 2, color, -1)
                
                # Draw motion pattern label
                cv2.putText(frame, trajectory.motion_pattern, 
                          (center[0] + 5, center[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except (ValueError, OverflowError) as e:
                logging.warning(f"Invalid prediction point: {str(e)}")
            
            return frame
        except Exception as e:
            logging.error(f"Error in draw_trajectory: {str(e)}")
            return frame
    
    def draw_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        try:
            # Normalize heatmap with exponential scaling for better visualization
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_norm = np.power(heatmap_norm, 0.5)  # Apply gamma correction
            heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Apply Gaussian blur for smoothing
            heatmap_colored = cv2.GaussianBlur(heatmap_colored, (7, 7), 0)
            
            # Apply adaptive alpha based on heatmap intensity
            max_intensity = np.max(heatmap_norm)
            alpha = self.heatmap_alpha * (1 + max_intensity / 255)  # Increase alpha for higher intensity
            
            return cv2.addWeighted(frame, 1 - alpha,
                                 heatmap_colored, alpha, 0)
        except Exception as e:
            logging.error(f"Error in draw_heatmap: {str(e)}")
            return frame
    
    def draw_interaction_lines(self, frame: np.ndarray, 
                             trajectories: Dict[int, ObjectTrajectory],
                             interaction_matrix: Dict[Tuple[str, str], int]) -> np.ndarray:
        try:
            for (class1, class2), count in interaction_matrix.items():
                if count > 0:
                    # Find objects of these classes
                    obj1 = next((t for t in trajectories.values() 
                               if t.class_id == class1), None)
                    obj2 = next((t for t in trajectories.values() 
                               if t.class_id == class2), None)
                    
                    if obj1 and obj2 and len(obj1.positions) > 0 and len(obj2.positions) > 0:
                        try:
                            pos1 = tuple(map(int, obj1.positions[-1]))
                            pos2 = tuple(map(int, obj2.positions[-1]))
                            
                            # Calculate interaction strength and color
                            strength = min(1.0, count / 100)  # Normalize to [0, 1]
                            color1 = self.color_map[class1]
                            color2 = self.color_map[class2]
                            color = tuple(int(c1 * (1 - strength) + c2 * strength)
                                       for c1, c2 in zip(color1, color2))
                            
                            # Draw line with varying thickness and alpha
                            thickness = max(1, int(3 * strength))
                            alpha = self.interaction_alpha * (0.3 + 0.7 * strength)
                            
                            # Create overlay for semi-transparent line
                            overlay = frame.copy()
                            cv2.line(overlay, pos1, pos2, color, thickness)
                            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
                            
                            # Draw interaction count with background
                            mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                            text = str(count)
                            font_scale = 0.5
                            thickness = 1
                            (text_width, text_height), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                            )
                            
                            # Draw background rectangle
                            padding = 2
                            cv2.rectangle(frame,
                                        (mid_point[0] - text_width//2 - padding,
                                         mid_point[1] - text_height - padding),
                                        (mid_point[0] + text_width//2 + padding,
                                         mid_point[1] + padding),
                                        (0, 0, 0), -1)
                            
                            # Draw text
                            cv2.putText(frame, text,
                                      (mid_point[0] - text_width//2,
                                       mid_point[1]),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, (255, 255, 255), thickness)
                        except (ValueError, OverflowError) as e:
                            logging.warning(f"Invalid interaction points: {str(e)}")
                            continue
            
            return frame
        except Exception as e:
            logging.error(f"Error in draw_interaction_lines: {str(e)}")
            return frame

    def annotate_frame(self, frame: np.ndarray, detections: Detections, model_names: List[str] = None) -> np.ndarray:
        """Safe wrapper for box annotation with comprehensive error handling"""
        if frame is None or frame.size == 0:
            return frame
            
        try:
            # Create a copy of the frame to avoid modifying the original
            annotated_frame = frame.copy()
            
            # Ensure detections are valid
            if len(detections) > 0:
                # Ensure all arrays in detections have the same length
                if not (len(detections.xyxy) == len(detections.confidence) == 
                       len(detections.class_id) == len(detections.tracker_id)):
                    logging.warning("Inconsistent detection arrays")
                    return frame
                
                # Prepare labels with error handling
                labels = []
                for i in range(len(detections)):
                    try:
                        class_id = int(detections.class_id[i])
                        confidence = float(detections.confidence[i])
                        tracker_id = int(detections.tracker_id[i])
                        
                        # Get class name from model_names if provided, otherwise use class_id
                        if model_names and 0 <= class_id < len(model_names):
                            class_name = model_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        
                        # Create label text
                        label = f"{class_name} {confidence:.2f} ID:{tracker_id}"
                        labels.append(label)
                    except Exception as e:
                        logging.warning(f"Error creating label for detection {i}: {str(e)}")
                        labels.append(f"Object {i}")
                
                # Draw boxes and labels manually with comprehensive error handling
                for i in range(len(detections)):
                    try:
                        # Get box coordinates with validation
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        
                        # Validate coordinates
                        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                            logging.warning(f"Invalid box coordinates for detection {i}")
                            continue
                        
                        # Get color for the box
                        class_id = int(detections.class_id[i])
                        color = self.color_map.get(class_id, (0, 255, 0))
                        
                        # Draw box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label = labels[i]
                        font_scale = 0.5
                        thickness = 1
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # Ensure text fits within frame
                        text_width = min(text_width, frame.shape[1] - x1 - 10)
                        
                        # Draw background rectangle
                        padding = 2
                        bg_y1 = max(0, y1 - text_height - padding)
                        bg_y2 = y1
                        bg_x1 = x1
                        bg_x2 = min(frame.shape[1], x1 + text_width + padding)
                        
                        # Draw background
                        cv2.rectangle(annotated_frame,
                                    (bg_x1, bg_y1),
                                    (bg_x2, bg_y2),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(annotated_frame, label,
                                  (x1 + padding, y1 - padding),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, (255, 255, 255), thickness)
                        
                    except Exception as e:
                        logging.warning(f"Error drawing box and label for detection {i}: {str(e)}")
                        continue
                
            return annotated_frame
            
        except Exception as e:
            logging.error(f"Error in frame annotation: {str(e)}")
            return frame  # Return original frame if annotation fails

class RealTimeObjectTracker:
    def __init__(self, mode: TrackingMode = TrackingMode.ADVANCED):
        self.mode = mode
        self.stats = TrackingStats()
        
        # Initialize YOLO model with confidence threshold and advanced settings
        self.model = YOLO('yolov8n.pt')
        self.conf_threshold = 0.5
        
        # Enable CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Configure tracking parameters
        self.tracker_config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8
        }
        
        # Initialize advanced visualizer
        self.visualizer = AdvancedVisualizer((1280, 720))
        
        # Initialize video capture with proper error handling
        self.cap = None
        self._initialize_camera()
        
        # Video writer for saving mode
        self.video_writer = None
        if mode == TrackingMode.SAVE_VIDEO:
            self._setup_video_writer()
            
        # Analytics storage
        self.analytics_data = []
        
        # Thread-safe queues for frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Initialize processing thread
        self.processing_thread = Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        logging.info(f"Tracker initialized in {mode.value} mode on {self.device}")

    def _initialize_camera(self):
        """Initialize camera with proper settings and error handling"""
        try:
            # Release existing camera if any
            if self.cap is not None:
                self.cap.release()
            
            # Initialize new camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video capture device")
            
            # Set camera properties with verification
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
            
            # Verify camera settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Camera initialized with resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Read multiple test frames to ensure camera is working properly
            for _ in range(5):  # Read 5 frames to warm up the camera
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    raise RuntimeError("Failed to read test frame from camera")
                
                # Check frame format
                if len(test_frame.shape) != 3 or test_frame.shape[2] != 3:
                    raise RuntimeError(f"Invalid frame format: {test_frame.shape}")
                
                # Check if frame is completely black
                if np.all(test_frame == 0):
                    raise RuntimeError("Camera is producing black frames")
                
                # Small delay to allow camera to stabilize
                time.sleep(0.1)
            
            logging.info("Camera initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            raise

    def _process_frames(self):
        while True:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break  # Exit if None is received
                
                # Ensure frame is valid and has correct shape
                if frame is None or frame.size == 0 or len(frame.shape) != 3:
                    logging.warning("Invalid frame received")
                    continue
                
                # Ensure frame is in correct format (BGR)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Run YOLO detection with tracking
                try:
                    # Ensure model is on correct device
                    if next(self.model.parameters()).device != self.device:
                        self.model.to(self.device)
                    
                    # Run inference with error handling
                    # Convert frame to numpy array if it's not already
                    if not isinstance(frame, np.ndarray):
                        frame = np.array(frame)
                    
                    # Run YOLO detection with tracking
                    results = self.model.track(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=0.5,
                        show=False,
                        verbose=False
                    )
                    
                    # Validate results
                    if not results or len(results) == 0:
                        logging.warning("No results from YOLO model")
                        self.result_queue.put((frame, Detections.empty()))
                        continue
                    
                    # Get first result and validate
                    result = results[0]
                    if not hasattr(result, 'boxes') or result.boxes is None:
                        logging.warning("No boxes in YOLO result")
                        self.result_queue.put((frame, Detections.empty()))
                        continue
                    
                    # Convert results to Detections format with validation
                    try:
                        if result.boxes.id is not None:
                            # Ensure all tensors are on CPU and convert to numpy
                            xyxy = result.boxes.xyxy.cpu().numpy()
                            confidence = result.boxes.conf.cpu().numpy()
                            class_id = result.boxes.cls.cpu().numpy().astype(int)
                            tracker_id = result.boxes.id.cpu().numpy().astype(int)
                            
                            # Validate array shapes
                            if not (len(xyxy) == len(confidence) == len(class_id) == len(tracker_id)):
                                logging.warning("Inconsistent detection arrays")
                                self.result_queue.put((frame, Detections.empty()))
                                continue
                            
                            # Create Detections object
                            tracked_detections = Detections(
                                xyxy=xyxy,
                                confidence=confidence,
                                class_id=class_id,
                                tracker_id=tracker_id
                            )
                        else:
                            tracked_detections = Detections.empty()
                        
                        # Put results in queue
                        self.result_queue.put((frame, tracked_detections))
                        
                    except (AttributeError, TypeError, ValueError) as e:
                        logging.error(f"Error converting YOLO results to Detections: {str(e)}")
                        self.result_queue.put((frame, Detections.empty()))
                    
                except Exception as e:
                    logging.error(f"Error in YOLO processing: {str(e)}")
                    # Put empty detections in case of error
                    self.result_queue.put((frame, Detections.empty()))
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in frame processing: {str(e)}")
                continue

    def _update_trajectories(self, detections: Detections, frame_time: float):
        current_time = time.time()
        
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            center = ((detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                     (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2)
            
            if track_id not in self.stats.trajectories:
                self.stats.trajectories[track_id] = ObjectTrajectory(
                    positions=[center],
                    velocities=[(0, 0)],
                    timestamps=[current_time],
                    class_id=detections.class_id[i],
                    track_id=track_id,
                    confidence_history=[detections.confidence[i]]
                )
            else:
                trajectory = self.stats.trajectories[track_id]
                last_pos = trajectory.positions[-1]
                velocity = (
                    (center[0] - last_pos[0]) / frame_time,
                    (center[1] - last_pos[1]) / frame_time
                )
                
                trajectory.positions.append(center)
                trajectory.velocities.append(velocity)
                trajectory.timestamps.append(current_time)
                trajectory.confidence_history.append(detections.confidence[i])
                
                # Keep only last 100 positions
                if len(trajectory.positions) > 100:
                    trajectory.positions.pop(0)
                    trajectory.velocities.pop(0)
                    trajectory.timestamps.pop(0)
                    trajectory.confidence_history.pop(0)
                
                # Update velocity statistics
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                self.stats.object_velocities[track_id].append(speed)

    def _update_heatmap(self, detections: Detections):
        # Update heatmap with current detections
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            self.stats.heatmap_data[y1:y2, x1:x2] += 1
        
        # Decay heatmap over time
        self.stats.heatmap_data *= 0.95

    def _update_interactions(self, detections: Detections):
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                # Calculate distance between objects
                center1 = ((detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                          (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2)
                center2 = ((detections.xyxy[j][0] + detections.xyxy[j][2]) / 2,
                          (detections.xyxy[j][1] + detections.xyxy[j][3]) / 2)
                
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 + 
                    (center1[1] - center2[1])**2
                )
                
                # If objects are close enough, record interaction
                if distance < 100:  # 100 pixels threshold
                    class1 = self.model.model.names[detections.class_id[i]]
                    class2 = self.model.model.names[detections.class_id[j]]
                    self.stats.interaction_matrix[(class1, class2)] += 1

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Detections]:
        try:
            start_time = time.time()
            
            # Validate input frame
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                logging.warning("Invalid input frame")
                return frame, Detections.empty()
            
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Add frame to processing queue if not full
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put(frame.copy(), timeout=0.1)  # Send a copy of the frame
                except queue.Full:
                    logging.warning("Frame queue is full, skipping frame")
                    return frame, Detections.empty()
            
            # Get results from processing thread with timeout
            try:
                processed_frame, detections = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                return frame, Detections.empty()
            
            # Update trajectories and analytics
            try:
                self._update_trajectories(detections, time.time() - start_time)
                self._update_heatmap(detections)
                self._update_interactions(detections)
                
                # Update motion patterns for all trajectories
                for trajectory in self.stats.trajectories.values():
                    trajectory.update_motion_pattern()
                
                # Update interaction history
                self.stats.update_interaction_history(time.time() - start_time)
            except Exception as e:
                logging.error(f"Error updating analytics: {str(e)}")
            
            # Use the safe annotation wrapper
            try:
                frame = self.visualizer.annotate_frame(frame, detections, self.model.model.names)
            except Exception as e:
                logging.error(f"Error in frame annotation: {str(e)}")
            
            # Draw trajectories and heatmap
            try:
                for trajectory in self.stats.trajectories.values():
                    frame = self.visualizer.draw_trajectory(frame, trajectory)
                
                frame = self.visualizer.draw_heatmap(frame, self.stats.heatmap_data)
                frame = self.visualizer.draw_interaction_lines(
                    frame,
                    self.stats.trajectories,
                    self.stats.interaction_matrix
                )
            except Exception as e:
                logging.error(f"Error in visualization: {str(e)}")
            
            # Update analytics if in analytics mode
            if self.mode == TrackingMode.ANALYTICS:
                try:
                    self._update_analytics(detections, time.time() - start_time)
                    interaction_stats = self.stats.get_interaction_statistics()
                    self._draw_interaction_stats(frame, interaction_stats)
                except Exception as e:
                    logging.error(f"Error updating analytics display: {str(e)}")
            
            return frame, detections
            
        except Exception as e:
            logging.error(f"Error in process_frame: {str(e)}")
            return frame, Detections.empty()

    def _draw_interaction_stats(self, frame: np.ndarray, stats: Dict):
        y_pos = 200  # Start position for interaction stats
        
        # Draw total interactions
        cv2.putText(
            frame,
            f"Total Interactions: {stats['total_interactions']}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_pos += 30
        
        # Draw most frequent interactions
        cv2.putText(
            frame,
            "Most Frequent:",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_pos += 30
        
        for interaction in stats['most_frequent']:
            text = f"{interaction['classes'][0]}-{interaction['classes'][1]}: {interaction['count']}"
            cv2.putText(
                frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_pos += 25
        
        # Draw longest duration interactions
        y_pos += 10
        cv2.putText(
            frame,
            "Longest Duration:",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_pos += 30
        
        for interaction in stats['longest_duration']:
            text = f"{interaction['classes'][0]}-{interaction['classes'][1]}: {interaction['duration']:.1f}s"
            cv2.putText(
                frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_pos += 25

    def _draw_advanced_stats(self, frame: np.ndarray):
        try:
            # Draw FPS and performance metrics with safety check
            if self.stats.fps_history:
                current_fps = self.stats.fps_history[-1]
            else:
                current_fps = 0.0
                
            cv2.putText(
                frame,
                f"FPS: {current_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Draw object counts and unique objects
            cv2.putText(
                frame,
                f"Objects: {len(self.stats.unique_objects)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Calculate and draw average velocity
            if self.stats.object_velocities:
                try:
                    avg_velocity = np.mean([
                        np.mean(velocities) for velocities in self.stats.object_velocities.values()
                        if velocities  # Only include non-empty lists
                    ])
                    cv2.putText(
                        frame,
                        f"Avg Velocity: {avg_velocity:.1f} px/s",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                except Exception as e:
                    logging.warning(f"Error calculating average velocity: {str(e)}")
                    cv2.putText(
                        frame,
                        "Avg Velocity: N/A",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
            
            # Draw predicted directions for active trajectories
            y_pos = 150
            cv2.putText(
                frame,
                "Predicted Directions:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_pos += 30
            
            # Get active trajectories (last 5 seconds)
            current_time = time.time()
            active_trajectories = [
                t for t in self.stats.trajectories.values()
                if t.timestamps and current_time - t.timestamps[-1] < 5.0
            ]
            
            # Sort trajectories by confidence and movement
            active_trajectories.sort(
                key=lambda t: (
                    np.mean(t.confidence_history[-5:]) if t.confidence_history else 0,
                    np.linalg.norm(t.velocities[-1]) if t.velocities else 0
                ),
                reverse=True
            )
            
            # Show top 3 most confident predictions
            for trajectory in active_trajectories[:3]:
                try:
                    if len(trajectory.positions) >= 2:
                        # Calculate direction vector using last 3 positions for smoother prediction
                        positions = trajectory.positions[-3:]
                        if len(positions) >= 2:
                            # Calculate average direction from last 3 positions
                            directions = []
                            for i in range(1, len(positions)):
                                direction = np.array(positions[i]) - np.array(positions[i-1])
                                if np.linalg.norm(direction) > 0:
                                    directions.append(direction)
                            
                            if directions:
                                # Average the directions
                                avg_direction = np.mean(directions, axis=0)
                                norm = np.linalg.norm(avg_direction)
                                
                                if norm > 0:
                                    avg_direction = avg_direction / norm
                                    
                                    # Convert to angle
                                    angle = np.arctan2(avg_direction[1], avg_direction[0])
                                    angle_deg = np.degrees(angle)
                                    
                                    # Determine cardinal direction with more precise angles
                                    if -22.5 <= angle_deg <= 22.5:
                                        direction_text = "→"
                                    elif 22.5 < angle_deg <= 67.5:
                                        direction_text = "↗"
                                    elif 67.5 < angle_deg <= 112.5:
                                        direction_text = "↑"
                                    elif 112.5 < angle_deg <= 157.5:
                                        direction_text = "↖"
                                    elif -157.5 <= angle_deg < -112.5:
                                        direction_text = "←"
                                    elif -112.5 <= angle_deg < -67.5:
                                        direction_text = "↙"
                                    elif -67.5 <= angle_deg < -22.5:
                                        direction_text = "↓"
                                    else:
                                        direction_text = "↘"
                                    
                                    # Get class name and speed
                                    class_name = self.model.model.names[trajectory.class_id]
                                    speed = np.linalg.norm(avg_direction) * 30  # Approximate speed in pixels per second
                                    
                                    # Draw prediction with speed
                                    cv2.putText(
                                        frame,
                                        f"{class_name}: {direction_text} ({speed:.1f} px/s)",
                                        (20, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 0),
                                        2
                                    )
                                    y_pos += 25
                except Exception as e:
                    logging.warning(f"Error calculating direction for trajectory: {str(e)}")
                    continue
            
            # Draw top 3 detected objects with confidence
            if self.stats.unique_objects:
                sorted_objects = sorted(
                    self.stats.unique_objects.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                y_pos = max(y_pos + 10, 250)  # Ensure minimum spacing
                cv2.putText(
                    frame,
                    "Top Detected Objects:",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                y_pos += 30
                
                for obj, count in sorted_objects:
                    cv2.putText(
                        frame,
                        f"{obj}: {count}",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y_pos += 25
                    
        except Exception as e:
            logging.error(f"Error in _draw_advanced_stats: {str(e)}")

    def _setup_video_writer(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            30.0,
            (1280, 720)
        )
        logging.info(f"Video writer initialized: {output_path}")

    def _update_analytics(self, detections: Detections, frame_time: float):
        frame_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_time": frame_time,
            "objects": []
        }
        
        for i in range(len(detections)):
            object_data = {
                "class": self.model.model.names[detections.class_id[i]],
                "confidence": float(detections.confidence[i]),
                "bbox": detections.xyxy[i].tolist(),
                "track_id": int(detections.tracker_id[i])
            }
            frame_data["objects"].append(object_data)
            
            # Update unique objects count
            class_name = self.model.model.names[detections.class_id[i]]
            self.stats.unique_objects[class_name] = self.stats.unique_objects.get(class_name, 0) + 1
            
        self.analytics_data.append(frame_data)
        self.stats.total_objects += len(detections)

    def _save_analytics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analytics_file = f"analytics_{timestamp}.json"
        
        analytics_summary = {
            "session_duration": time.time() - self.stats.start_time,
            "total_frames": self.stats.frame_count,
            "average_fps": np.mean(self.stats.fps_history),
            "unique_objects": self.stats.unique_objects,
            "total_objects": self.stats.total_objects,
            "frame_data": self.analytics_data
        }
        
        with open(analytics_file, 'w') as f:
            json.dump(analytics_summary, f, indent=2)
            
        logging.info(f"Analytics saved to {analytics_file}")

    def run(self):
        try:
            fps_update_interval = time.time()
            frame_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            while True:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        logging.error(f"Failed to read frame from camera (attempt {consecutive_failures}/{max_consecutive_failures})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logging.error("Too many consecutive frame capture failures. Exiting...")
                            break
                        
                        # Try to reinitialize camera
                        self._initialize_camera()
                        continue
                    
                    # Reset consecutive failures on successful frame capture
                    consecutive_failures = 0
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        logging.warning("Invalid frame received from camera")
                        continue
                    
                    # Check if frame is completely black
                    if np.all(frame == 0):
                        logging.error("Detected black frame, reinitializing camera")
                        self._initialize_camera()
                        continue
                    
                    # Ensure frame is in correct format (BGR)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Process frame
                    frame, detections = self.process_frame(frame)
                    
                    # Update FPS
                    frame_count += 1
                    current_time = time.time()
                    if current_time - fps_update_interval >= 1.0:
                        current_fps = frame_count / (current_time - fps_update_interval)
                        self.stats.fps_history.append(current_fps)
                        fps_update_interval = current_time
                        frame_count = 0
                        
                        if len(self.stats.fps_history) > 10:
                            avg_fps = np.mean(self.stats.fps_history[-10:])
                            if avg_fps < 15:
                                logging.warning(f"Low FPS detected: {avg_fps:.2f}")
                    
                    # Draw advanced statistics
                    try:
                        self._draw_advanced_stats(frame)
                    except Exception as e:
                        logging.error(f"Error drawing advanced stats: {str(e)}")
                    
                    # Save frame if in video mode
                    if self.mode == TrackingMode.SAVE_VIDEO and self.video_writer is not None:
                        try:
                            self.video_writer.write(frame)
                        except Exception as e:
                            logging.error(f"Error writing frame to video: {str(e)}")
                    
                    # Show frame
                    try:
                        cv2.imshow("Advanced Real-Time Object Detection & Tracking", frame)
                    except Exception as e:
                        logging.error(f"Error displaying frame: {str(e)}")
                    
                    # Break loop on 'q' press or window close
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or cv2.getWindowProperty("Advanced Real-Time Object Detection & Tracking", cv2.WND_PROP_VISIBLE) < 1:
                        break
                        
                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    continue
                
        finally:
            # Clean up
            logging.info("Cleaning up resources...")
            
            # Stop processing thread
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.frame_queue.put(None)  # Signal to stop
                self.processing_thread.join(timeout=1.0)
            
            # Release video resources
            if self.cap is not None:
                self.cap.release()
            
            if self.video_writer is not None:
                self.video_writer.release()
            
            # Close all windows
            cv2.destroyAllWindows()
            
            # Save analytics if needed
            if self.mode == TrackingMode.ANALYTICS:
                try:
                    self._save_analytics()
                except Exception as e:
                    logging.error(f"Failed to save analytics: {str(e)}")
            
            logging.info("Tracking session ended")

if __name__ == "__main__":
    # Example usage with advanced mode
    tracker = RealTimeObjectTracker(mode=TrackingMode.ADVANCED)
    tracker.run() 