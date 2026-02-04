"""
STATUS: Active - Core Configuration
PURPOSE: Central configuration management for Robot Referee system
NEXT_STEPS: 
  - Adjust paths based on deployment environment
  - Add model hyperparameters as needed
  - Configure case-specific thresholds
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Directory paths
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"
SRC_DIR = ROOT_DIR / "src"

# Model configuration
YOLO_POSE_MODEL = "yolov8n-pose.pt"  # Lightweight model for pose detection
YOLO_DETECT_MODEL = "yolov8n.pt"      # Lightweight model for object detection

# Inference settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = "cuda"  # Use "cpu" if no GPU available

# Case-specific configurations
CASES_CONFIG = {
    "unnatural_arm": {
        "enabled": True,
        "angle_threshold": 135,  # Degrees - arms above this angle are unnatural
        "min_confidence": 0.6
    },
    "reaction_time": {
        "enabled": True,
        "max_time_ms": 500,  # Maximum reaction time in milliseconds
        "tracking_fps": 30
    },
    "deflection": {
        "enabled": True,
        "min_trajectory_change": 15,  # Degrees - minimum angle change to consider deflection
        "ball_confidence": 0.7
    }
}

# Training configuration (for Colab)
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "patience": 10,  # Early stopping patience
    "mixed_precision": True,  # AMP for faster training
    "workers": 8,
    "project": "robot_referee",
    "name": "training_run"
}

# Video processing
VIDEO_CONFIG = {
    "output_fps": 30,
    "output_codec": "mp4v",
    "draw_annotations": True
}

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATASET_DIR, MODELS_DIR, CONFIGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Models directory: {MODELS_DIR}")
