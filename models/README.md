# Models Directory

This directory stores trained YOLOv8 model weights.

## Structure

```
models/
├── yolov8n-pose.pt       # Pre-trained pose model (downloaded)
├── yolov8n.pt            # Pre-trained detection model (downloaded)
├── custom_pose.pt        # Your trained pose model
└── custom_detect.pt      # Your trained detection model
```

## Model Files

### Pre-trained Models

Download pre-trained YOLOv8 models:
- Pose: `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`
- Detection: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`

Models will be automatically downloaded by Ultralytics on first use.

### Custom Trained Models

After training in `train.ipynb`, copy your best model here:
```bash
cp robot_referee/training_run/weights/best.pt models/custom_pose.pt
```

## Model Selection

Choose based on your requirements:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| n (nano) | ⚡⚡⚡ | ⭐⭐ | Real-time, edge devices |
| s (small) | ⚡⚡ | ⭐⭐⭐ | Balanced |
| m (medium) | ⚡ | ⭐⭐⭐⭐ | High accuracy |
| l (large) | 🐌 | ⭐⭐⭐⭐⭐ | Maximum accuracy |

## Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO("models/yolov8n-pose.pt")

# Run inference
results = model("video.mp4")
```
