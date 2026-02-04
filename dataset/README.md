# Dataset Directory

Place your dataset files here following the structure:

```
dataset/
├── data.yaml              # Dataset configuration file
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images (optional)
└── labels/
    ├── train/            # Training labels in YOLO format
    ├── val/              # Validation labels
    └── test/             # Test labels (optional)
```

## YOLO Format

For object detection and pose estimation, labels should be in YOLO format:

### Object Detection Format (labels/*.txt)
```
<class_id> <x_center> <y_center> <width> <height>
```

### Pose Detection Format (labels/*.txt)
```
<class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_vis> ... <kpt17_x> <kpt17_y> <kpt17_vis>
```

All coordinates should be normalized to [0, 1].

## Data Preparation Tips

1. Ensure consistent image quality
2. Include diverse lighting conditions
3. Vary camera angles and distances
4. Balance classes in training data
5. Augment data if dataset is small

## Data Sources

Potential sources for football video data:
- Public sports video datasets
- Licensed match footage
- Synthetic/simulated data
- Open-source sports archives

**Note:** Ensure you have proper rights/licenses for any data you use.
