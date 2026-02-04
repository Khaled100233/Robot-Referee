"""
STATUS: Active - Reaction Time Analysis
PURPOSE: Detect handball cases based on player reaction time to ball movement
         by tracking ball trajectory and player response
NEXT_STEPS:
  - Implement ball tracking with Kalman filter
  - Add player attention/gaze estimation
  - Calibrate reaction time thresholds with real game data
"""

import numpy as np
from collections import deque


class ReactionTimeDetector:
    """Detector for handball based on insufficient reaction time."""
    
    def __init__(self, config):
        """
        Initialize Reaction Time detector.
        
        Args:
            config: Configuration dictionary with max_time_ms, tracking_fps
        """
        self.max_time_ms = config.get("max_time_ms", 500)
        self.tracking_fps = config.get("tracking_fps", 30)
        
        # Calculate max frames for reaction time
        self.max_reaction_frames = int(self.max_time_ms * self.tracking_fps / 1000)
        
        # Track ball positions over time
        self.ball_history = deque(maxlen=60)  # Last 2 seconds at 30fps
        
        # Track player positions
        self.player_history = deque(maxlen=60)
        
        # Track contact events
        self.contact_events = []
    
    def detect(self, frame, pose_results, detect_results):
        """
        Detect handball based on reaction time analysis.
        
        Args:
            frame: Input frame (BGR image)
            pose_results: YOLO pose detection results
            detect_results: YOLO object detection results
            
        Returns:
            dict: Detection result with detected flag and details
        """
        result = {
            "detected": False,
            "reaction_time_ms": None,
            "details": []
        }
        
        # Extract ball position from detections
        ball_pos = self._extract_ball_position(detect_results)
        if ball_pos is not None:
            self.ball_history.append(ball_pos)
        
        # Extract player positions
        player_positions = self._extract_player_positions(pose_results)
        self.player_history.append(player_positions)
        
        # Need sufficient history to analyze
        if len(self.ball_history) < self.max_reaction_frames:
            return result
        
        # Analyze ball trajectory changes
        trajectory_change = self._detect_trajectory_change()
        
        if trajectory_change:
            # Check if any player could have reacted in time
            reaction_analysis = self._analyze_reaction_time(trajectory_change)
            
            if reaction_analysis["insufficient_time"]:
                result["detected"] = True
                result["reaction_time_ms"] = reaction_analysis["reaction_time_ms"]
                result["details"].append(
                    f"Insufficient reaction time: {reaction_analysis['reaction_time_ms']:.0f}ms"
                )
        
        return result
    
    def _extract_ball_position(self, detect_results):
        """
        Extract ball position from detection results.
        
        Args:
            detect_results: YOLO detection results
            
        Returns:
            tuple: (x, y) ball position or None
        """
        if not detect_results or len(detect_results) == 0:
            return None
        
        # Look for ball in detections (class 32 is sports ball in COCO)
        for box in detect_results[0].boxes:
            # Check if it's a ball (you may need to adjust class ID)
            # For now, we'll assume the smallest circular object is the ball
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Simple heuristic: ball should be relatively small
            width = x2 - x1
            height = y2 - y1
            if width < 100 and height < 100:
                return (center_x, center_y)
        
        return None
    
    def _extract_player_positions(self, pose_results):
        """
        Extract player center positions from pose results.
        
        Args:
            pose_results: YOLO pose results
            
        Returns:
            list: List of (x, y) player positions
        """
        positions = []
        
        if not pose_results or len(pose_results) == 0:
            return positions
        
        if len(pose_results[0].keypoints.data) == 0:
            return positions
        
        for keypoints in pose_results[0].keypoints.data:
            keypoints_np = keypoints.cpu().numpy()
            # Use torso keypoints for player center (shoulders and hips)
            valid_points = keypoints_np[keypoints_np[:, 2] > 0.5][:, :2]
            
            if len(valid_points) > 0:
                center = valid_points.mean(axis=0)
                positions.append(tuple(center))
        
        return positions
    
    def _detect_trajectory_change(self):
        """
        Detect significant changes in ball trajectory.
        
        Returns:
            dict: Information about trajectory change or None
        """
        if len(self.ball_history) < 10:
            return None
        
        # Calculate velocity vectors
        positions = list(self.ball_history)
        velocities = []
        
        for i in range(1, len(positions)):
            vel = (
                positions[i][0] - positions[i-1][0],
                positions[i][1] - positions[i-1][1]
            )
            velocities.append(vel)
        
        if len(velocities) < 5:
            return None
        
        # Check for sudden direction change
        recent_vel = np.array(velocities[-3:]).mean(axis=0)
        prev_vel = np.array(velocities[-10:-5]).mean(axis=0)
        
        # Calculate angle between velocity vectors
        dot_product = np.dot(recent_vel, prev_vel)
        norms = np.linalg.norm(recent_vel) * np.linalg.norm(prev_vel)
        
        if norms < 1e-6:
            return None
        
        cos_angle = dot_product / norms
        angle_change = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Significant trajectory change detected
        if angle_change > 30:  # More than 30 degrees change
            return {
                "frame": len(self.ball_history),
                "angle_change": angle_change,
                "position": positions[-1]
            }
        
        return None
    
    def _analyze_reaction_time(self, trajectory_change):
        """
        Analyze if player had sufficient reaction time.
        
        Args:
            trajectory_change: Dictionary with trajectory change info
            
        Returns:
            dict: Analysis results
        """
        # Calculate time from ball direction change to contact
        # This is a simplified version - would need more sophisticated tracking
        
        # Estimate reaction time based on frame count
        # In production, this would track actual frames from trajectory change to contact
        estimated_frames_to_contact = len(self.ball_history) - trajectory_change["frame"]
        reaction_time_ms = estimated_frames_to_contact * (1000 / self.tracking_fps)
        
        return {
            "insufficient_time": reaction_time_ms < self.max_time_ms,
            "reaction_time_ms": reaction_time_ms
        }
