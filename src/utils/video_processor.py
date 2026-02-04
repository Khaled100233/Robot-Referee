"""
STATUS: Active - Video Processing Utilities
PURPOSE: Handle video input/output operations and frame management
NEXT_STEPS:
  - Add multi-threaded video processing for performance
  - Support live camera input
  - Add frame skip/sample functionality for long videos
"""

import cv2
from pathlib import Path


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self):
        """Initialize video processor."""
        pass
    
    def read_video(self, video_path):
        """
        Read a video file and yield frames.
        
        Args:
            video_path: Path to video file
            
        Yields:
            tuple: (frame_number, frame)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                yield frame_count, frame
        finally:
            cap.release()
    
    def get_video_info(self, video_path):
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information (fps, width, height, total_frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return info
    
    def save_frame(self, frame, output_path):
        """
        Save a single frame as an image.
        
        Args:
            frame: Frame to save
            output_path: Output file path
        """
        cv2.imwrite(str(output_path), frame)
    
    def create_video_writer(self, output_path, fps, width, height, codec="mp4v"):
        """
        Create a video writer object.
        
        Args:
            output_path: Output video path
            fps: Frames per second
            width: Video width
            height: Video height
            codec: Video codec (default: mp4v)
            
        Returns:
            cv2.VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        return writer
