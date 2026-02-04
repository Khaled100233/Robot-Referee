"""
STATUS: Active - Visualization Utilities
PURPOSE: Draw detection results and annotations on video frames
NEXT_STEPS:
  - Add customizable color schemes
  - Implement bounding box smoothing
  - Add trajectory visualization
"""

import cv2
import numpy as np


class Visualizer:
    """Utility class for visualizing detection results."""
    
    def __init__(self):
        """Initialize visualizer with color schemes."""
        # Color palette (BGR format for OpenCV)
        self.colors = {
            "handball": (0, 0, 255),      # Red
            "normal": (0, 255, 0),         # Green
            "warning": (0, 165, 255),      # Orange
            "info": (255, 255, 0),         # Cyan
            "pose": (255, 0, 255),         # Magenta
            "ball": (0, 255, 255)          # Yellow
        }
    
    def draw_results(self, frame, case_results, pose_results, detect_results):
        """
        Draw all detection results on frame.
        
        Args:
            frame: Input frame (BGR)
            case_results: Dictionary of case detection results
            pose_results: YOLO pose detection results
            detect_results: YOLO object detection results
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        # Draw pose keypoints
        if pose_results and len(pose_results) > 0:
            self._draw_poses(frame, pose_results)
        
        # Draw object detections (ball, etc.)
        if detect_results and len(detect_results) > 0:
            self._draw_detections(frame, detect_results)
        
        # Draw case-specific annotations
        self._draw_case_results(frame, case_results)
        
        return frame
    
    def _draw_poses(self, frame, pose_results):
        """Draw pose keypoints and skeleton."""
        if len(pose_results[0].keypoints.data) == 0:
            return
        
        for keypoints in pose_results[0].keypoints.data:
            keypoints_np = keypoints.cpu().numpy()
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints_np):
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, self.colors["pose"], -1)
            
            # Draw skeleton connections (simplified)
            self._draw_skeleton(frame, keypoints_np)
    
    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections between keypoints."""
        # Define skeleton connections (COCO format)
        skeleton = [
            (5, 7), (7, 9),    # Left arm
            (6, 8), (8, 10),   # Right arm
            (5, 6),            # Shoulders
            (5, 11), (6, 12),  # Torso
            (11, 12),          # Hips
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        for start_idx, end_idx in skeleton:
            if (keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5):
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                cv2.line(frame, start_point, end_point, self.colors["pose"], 2)
    
    def _draw_detections(self, frame, detect_results):
        """Draw object detection bounding boxes."""
        for box in detect_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                self.colors["ball"], 
                2
            )
            
            # Draw confidence
            label = f"Ball: {conf:.2f}"
            cv2.putText(
                frame, label, 
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                self.colors["ball"], 
                2
            )
    
    def _draw_case_results(self, frame, case_results):
        """Draw case-specific detection results."""
        y_offset = 30
        
        for case_name, result in case_results.items():
            if result.get("detected", False):
                # Draw warning banner
                text = f"HANDBALL - {case_name.upper().replace('_', ' ')}"
                
                # Draw background rectangle
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )[0]
                cv2.rectangle(
                    frame, 
                    (10, y_offset - 25), 
                    (text_size[0] + 20, y_offset + 10),
                    self.colors["handball"], 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame, text,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )
                
                # Draw details
                y_offset += 50
                for detail in result.get("details", []):
                    cv2.putText(
                        frame, detail,
                        (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self.colors["warning"],
                        2
                    )
                    y_offset += 30
    
    def draw_trajectory(self, frame, positions, color=None):
        """
        Draw trajectory line on frame.
        
        Args:
            frame: Input frame
            positions: List of (x, y) positions
            color: Line color (BGR), defaults to yellow
        """
        if color is None:
            color = self.colors["ball"]
        
        if len(positions) < 2:
            return
        
        for i in range(1, len(positions)):
            pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(frame, pt1, pt2, color, 2)
