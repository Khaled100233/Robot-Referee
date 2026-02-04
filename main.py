"""
STATUS: Active - Main Inference Entry Point
PURPOSE: Local inference script for Robot Referee cases on video footage
NEXT_STEPS:
  - Add command-line arguments for video input/output
  - Implement real-time video processing
  - Add visualization overlays for detected cases
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

from config import (
    YOLO_POSE_MODEL, YOLO_DETECT_MODEL, MODELS_DIR, 
    CONFIDENCE_THRESHOLD, CASES_CONFIG, VIDEO_CONFIG
)
from src.cases.unnatural_arm import UnnaturalArmDetector
from src.cases.reaction_time import ReactionTimeDetector
from src.cases.deflection import DeflectionDetector
from src.utils.video_processor import VideoProcessor
from src.utils.visualizer import Visualizer


class RobotReferee:
    """Main class for Robot Referee inference system."""
    
    def __init__(self, pose_model_path=None, detect_model_path=None):
        """
        Initialize Robot Referee system.
        
        Args:
            pose_model_path: Path to YOLOv8 pose model
            detect_model_path: Path to YOLOv8 detection model
        """
        print("Initializing Robot Referee...")
        
        # Load models
        self.pose_model_path = pose_model_path or MODELS_DIR / YOLO_POSE_MODEL
        self.detect_model_path = detect_model_path or MODELS_DIR / YOLO_DETECT_MODEL
        
        print(f"Loading pose model: {self.pose_model_path}")
        self.pose_model = YOLO(str(self.pose_model_path))
        
        print(f"Loading detection model: {self.detect_model_path}")
        self.detect_model = YOLO(str(self.detect_model_path))
        
        # Initialize case detectors
        self.detectors = {}
        if CASES_CONFIG["unnatural_arm"]["enabled"]:
            self.detectors["unnatural_arm"] = UnnaturalArmDetector(
                CASES_CONFIG["unnatural_arm"]
            )
        if CASES_CONFIG["reaction_time"]["enabled"]:
            self.detectors["reaction_time"] = ReactionTimeDetector(
                CASES_CONFIG["reaction_time"]
            )
        if CASES_CONFIG["deflection"]["enabled"]:
            self.detectors["deflection"] = DeflectionDetector(
                CASES_CONFIG["deflection"]
            )
        
        # Initialize utilities
        self.video_processor = VideoProcessor()
        self.visualizer = Visualizer()
        
        print("Robot Referee initialized successfully!")
    
    def process_frame(self, frame):
        """
        Process a single frame for all enabled cases.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            dict: Detection results for each case
        """
        results = {}
        
        # Run pose detection
        pose_results = self.pose_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Run object detection (for ball, players, etc.)
        detect_results = self.detect_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Process each case
        for case_name, detector in self.detectors.items():
            case_result = detector.detect(frame, pose_results, detect_results)
            results[case_name] = case_result
        
        return results, pose_results, detect_results
    
    def process_video(self, input_path, output_path=None):
        """
        Process a video file and detect handball cases.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
        """
        print(f"\nProcessing video: {input_path}")
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["output_codec"])
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
        
        frame_count = 0
        detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                results, pose_results, detect_results = self.process_frame(frame)
                
                # Store detections
                detections.append({
                    "frame": frame_count,
                    "results": results
                })
                
                # Visualize if output is requested
                if writer:
                    annotated_frame = self.visualizer.draw_results(
                        frame.copy(), results, pose_results, detect_results
                    )
                    writer.write(annotated_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames", end="\r")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        print(f"\nCompleted! Processed {frame_count} frames")
        self._print_summary(detections)
        
        return detections
    
    def _print_summary(self, detections):
        """Print summary of detections."""
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        
        for case_name in self.detectors.keys():
            detected_frames = [
                d for d in detections 
                if d["results"].get(case_name, {}).get("detected", False)
            ]
            print(f"{case_name.upper()}: {len(detected_frames)} detections")
        
        print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robot Referee - Handball Detection System"
    )
    parser.add_argument(
        "--video", "-v", required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o", help="Path to output video file (optional)"
    )
    parser.add_argument(
        "--pose-model", help="Path to custom pose model"
    )
    parser.add_argument(
        "--detect-model", help="Path to custom detection model"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    referee = RobotReferee(
        pose_model_path=args.pose_model,
        detect_model_path=args.detect_model
    )
    
    # Process video
    referee.process_video(
        input_path=args.video,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
