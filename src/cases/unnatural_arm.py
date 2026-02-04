"""
STATUS: Active - Unnatural Arm Position Detection
PURPOSE: Detect handball cases where player's arm is in an unnatural position
         using geometric analysis of pose keypoints
NEXT_STEPS:
  - Fine-tune angle thresholds based on football rules
  - Add body orientation analysis
  - Implement temporal smoothing to reduce false positives
"""

import numpy as np
import math


class UnnaturalArmDetector:
    """Detector for unnatural arm positions that may indicate handball."""
    
    def __init__(self, config):
        """
        Initialize Unnatural Arm detector.
        
        Args:
            config: Configuration dictionary with angle_threshold, min_confidence
        """
        self.angle_threshold = config.get("angle_threshold", 135)
        self.min_confidence = config.get("min_confidence", 0.6)
    
    def detect(self, frame, pose_results, detect_results):
        """
        Detect unnatural arm positions in the frame.
        
        Args:
            frame: Input frame (BGR image)
            pose_results: YOLO pose detection results
            detect_results: YOLO object detection results
            
        Returns:
            dict: Detection result with detected flag and details
        """
        result = {
            "detected": False,
            "players": [],
            "details": []
        }
        
        if not pose_results or len(pose_results) == 0:
            return result
        
        # Process each detected person
        for detection in pose_results[0].boxes:
            if len(pose_results[0].keypoints.data) == 0:
                continue
            
            # Get pose keypoints for this detection
            # YOLOv8 pose keypoints: 0-nose, 5-left_shoulder, 6-right_shoulder,
            # 7-left_elbow, 8-right_elbow, 9-left_wrist, 10-right_wrist
            for keypoints in pose_results[0].keypoints.data:
                arm_angles = self._calculate_arm_angles(keypoints)
                
                if arm_angles:
                    is_unnatural, details = self._is_unnatural_position(arm_angles)
                    
                    if is_unnatural:
                        result["detected"] = True
                        result["players"].append({
                            "keypoints": keypoints.cpu().numpy(),
                            "angles": arm_angles
                        })
                        result["details"].append(details)
        
        return result
    
    def _calculate_arm_angles(self, keypoints):
        """
        Calculate arm angles from pose keypoints.
        
        Args:
            keypoints: Tensor of shape (17, 3) with (x, y, confidence)
            
        Returns:
            dict: Calculated angles for left and right arms
        """
        # Extract keypoints (x, y, confidence)
        keypoints = keypoints.cpu().numpy()
        
        # Check if required keypoints are detected
        required_indices = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
        if not all(keypoints[i, 2] > self.min_confidence for i in required_indices):
            return None
        
        # Calculate angles
        left_shoulder = keypoints[5, :2]
        right_shoulder = keypoints[6, :2]
        left_elbow = keypoints[7, :2]
        right_elbow = keypoints[8, :2]
        left_wrist = keypoints[9, :2]
        right_wrist = keypoints[10, :2]
        
        # Calculate arm angles relative to body
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Calculate elevation angles (angle from horizontal)
        left_elevation = self._calculate_elevation(left_shoulder, left_elbow)
        right_elevation = self._calculate_elevation(right_shoulder, right_elbow)
        
        return {
            "left_arm_angle": left_angle,
            "right_arm_angle": right_angle,
            "left_elevation": left_elevation,
            "right_elevation": right_elevation
        }
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points.
        
        Args:
            point1, point2, point3: (x, y) coordinates
            
        Returns:
            float: Angle in degrees
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_elevation(self, shoulder, elbow):
        """
        Calculate elevation angle of arm from horizontal.
        
        Args:
            shoulder, elbow: (x, y) coordinates
            
        Returns:
            float: Elevation angle in degrees
        """
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        
        angle = np.degrees(np.arctan2(-dy, dx))  # Negative because y increases downward
        return angle
    
    def _is_unnatural_position(self, arm_angles):
        """
        Determine if arm position is unnatural based on angles.
        
        Args:
            arm_angles: Dictionary of calculated angles
            
        Returns:
            tuple: (is_unnatural: bool, details: str)
        """
        # Check if either arm has unusual elevation
        left_elevated = abs(arm_angles["left_elevation"]) > self.angle_threshold - 45
        right_elevated = abs(arm_angles["right_elevation"]) > self.angle_threshold - 45
        
        if left_elevated or right_elevated:
            side = "left" if left_elevated else "right"
            elevation = arm_angles[f"{side}_elevation"]
            return True, f"Unnatural {side} arm elevation: {elevation:.1f}°"
        
        # Check for extended arms (potential handball position)
        left_extended = arm_angles["left_arm_angle"] > self.angle_threshold
        right_extended = arm_angles["right_arm_angle"] > self.angle_threshold
        
        if left_extended or right_extended:
            side = "left" if left_extended else "right"
            angle = arm_angles[f"{side}_arm_angle"]
            return True, f"Unnatural {side} arm extension: {angle:.1f}°"
        
        return False, ""
