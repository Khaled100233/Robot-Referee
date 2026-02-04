"""
STATUS: Active - Ball Deflection Detection
PURPOSE: Detect handball by analyzing ball trajectory deflections when near players
NEXT_STEPS:
  - Implement robust ball tracking with Kalman filter
  - Add trajectory prediction for expected vs actual path
  - Integrate with player pose to determine contact point
"""

import numpy as np
from collections import deque


class DeflectionDetector:
    """Detector for handball based on ball deflection analysis."""
    
    def __init__(self, config):
        """
        Initialize Deflection detector.
        
        Args:
            config: Configuration with min_trajectory_change, ball_confidence
        """
        self.min_trajectory_change = config.get("min_trajectory_change", 15)
        self.ball_confidence = config.get("ball_confidence", 0.7)
        
        # Track ball trajectory
        self.ball_positions = deque(maxlen=90)  # 3 seconds at 30fps
        self.ball_velocities = deque(maxlen=90)
        
        # Track players near ball
        self.player_proximity = deque(maxlen=90)
    
    def detect(self, frame, pose_results, detect_results):
        """
        Detect deflection by analyzing ball trajectory changes near players.
        
        Args:
            frame: Input frame (BGR image)
            pose_results: YOLO pose detection results
            detect_results: YOLO object detection results
            
        Returns:
            dict: Detection result with detected flag and details
        """
        result = {
            "detected": False,
            "deflection_angle": None,
            "contact_player": None,
            "details": []
        }
        
        # Extract ball position
        ball_info = self._extract_ball_info(detect_results)
        if ball_info is None:
            return result
        
        self.ball_positions.append(ball_info["position"])
        
        # Calculate velocity
        if len(self.ball_positions) >= 2:
            velocity = self._calculate_velocity(
                self.ball_positions[-2], 
                self.ball_positions[-1]
            )
            self.ball_velocities.append(velocity)
        
        # Extract player positions
        player_positions = self._extract_player_positions(pose_results)
        self.player_proximity.append(player_positions)
        
        # Need sufficient history
        if len(self.ball_positions) < 10 or len(self.ball_velocities) < 5:
            return result
        
        # Detect deflection
        deflection = self._detect_deflection()
        
        if deflection:
            # Check if deflection occurred near a player
            nearby_player = self._find_nearby_player(deflection["position"])
            
            if nearby_player is not None:
                result["detected"] = True
                result["deflection_angle"] = deflection["angle"]
                result["contact_player"] = nearby_player
                result["details"].append(
                    f"Ball deflection detected: {deflection['angle']:.1f}° change"
                )
        
        return result
    
    def _extract_ball_info(self, detect_results):
        """
        Extract ball information from detection results.
        
        Args:
            detect_results: YOLO detection results
            
        Returns:
            dict: Ball information with position, confidence
        """
        if not detect_results or len(detect_results) == 0:
            return None
        
        # Look for ball in detections
        for box in detect_results[0].boxes:
            conf = float(box.conf[0])
            
            if conf < self.ball_confidence:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            
            # Ball should be relatively small and circular
            if width < 100 and height < 100:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                return {
                    "position": (center_x, center_y),
                    "confidence": conf,
                    "size": (width, height)
                }
        
        return None
    
    def _extract_player_positions(self, pose_results):
        """
        Extract player positions from pose results.
        
        Args:
            pose_results: YOLO pose results
            
        Returns:
            list: List of player position dictionaries
        """
        positions = []
        
        if not pose_results or len(pose_results) == 0:
            return positions
        
        if len(pose_results[0].keypoints.data) == 0:
            return positions
        
        for idx, keypoints in enumerate(pose_results[0].keypoints.data):
            keypoints_np = keypoints.cpu().numpy()
            
            # Get valid keypoints
            valid_mask = keypoints_np[:, 2] > 0.5
            valid_points = keypoints_np[valid_mask][:, :2]
            
            if len(valid_points) > 0:
                center = valid_points.mean(axis=0)
                positions.append({
                    "id": idx,
                    "center": tuple(center),
                    "keypoints": keypoints_np
                })
        
        return positions
    
    def _calculate_velocity(self, pos1, pos2):
        """
        Calculate velocity vector between two positions.
        
        Args:
            pos1, pos2: (x, y) positions
            
        Returns:
            tuple: (vx, vy) velocity vector
        """
        return (pos2[0] - pos1[0], pos2[1] - pos1[1])
    
    def _detect_deflection(self):
        """
        Detect significant deflection in ball trajectory.
        
        Returns:
            dict: Deflection information or None
        """
        if len(self.ball_velocities) < 10:
            return None
        
        velocities = list(self.ball_velocities)
        
        # Compare recent velocity with previous velocity
        recent_vel = np.array(velocities[-3:]).mean(axis=0)
        prev_vel = np.array(velocities[-10:-5]).mean(axis=0)
        
        # Calculate angle change
        dot_product = np.dot(recent_vel, prev_vel)
        norms = np.linalg.norm(recent_vel) * np.linalg.norm(prev_vel)
        
        if norms < 1e-6:
            return None
        
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle_change = np.degrees(np.arccos(cos_angle))
        
        # Check for significant deflection
        if angle_change > self.min_trajectory_change:
            positions = list(self.ball_positions)
            return {
                "angle": angle_change,
                "position": positions[-1],
                "velocity_before": prev_vel,
                "velocity_after": recent_vel
            }
        
        return None
    
    def _find_nearby_player(self, ball_position, proximity_threshold=150):
        """
        Find player near the ball position.
        
        Args:
            ball_position: (x, y) ball position
            proximity_threshold: Maximum distance in pixels
            
        Returns:
            dict: Player information or None
        """
        if len(self.player_proximity) == 0:
            return None
        
        recent_players = self.player_proximity[-1]
        
        for player in recent_players:
            distance = np.linalg.norm(
                np.array(player["center"]) - np.array(ball_position)
            )
            
            if distance < proximity_threshold:
                return {
                    "id": player["id"],
                    "distance": distance,
                    "center": player["center"]
                }
        
        return None
