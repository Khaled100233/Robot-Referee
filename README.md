# Robot Referee

**STATUS:** Active - Production Ready  
**PURPOSE:** Automated handball detection system for football using Computer Vision  
**NEXT_STEPS:** Deploy to production, collect real-world data, fine-tune models

---

A sophisticated computer vision system that automatically detects handball infractions in football (soccer) videos using YOLOv8 Pose and Object Detection models.

## 🎯 Features

### Three Detection Cases

1. **Unnatural Arm Position (Geometry-Based)**
   - Analyzes player arm angles and positions using pose estimation
   - Detects arms in unnatural positions that increase body surface area
   - Uses geometric calculations to determine arm elevation and extension angles

2. **Reaction Time Analysis**
   - Tracks ball trajectory and player movements
   - Analyzes whether player had sufficient time to react
   - Detects cases where ball contact was unavoidable

3. **Deflection Detection**
   - Monitors ball trajectory changes near players
   - Detects significant deflections indicating contact
   - Correlates trajectory changes with player proximity

## 🏗️ Project Structure

```
Robot-Referee/
├── main.py                    # Local inference entry point
├── train.ipynb                # Colab training notebook with optimizations
├── config.py                  # Central configuration management
├── requirements.txt           # Python dependencies
├── src/
│   ├── cases/                 # Detection case implementations
│   │   ├── unnatural_arm.py   # Unnatural arm position detector
│   │   ├── reaction_time.py   # Reaction time analyzer
│   │   └── deflection.py      # Ball deflection detector
│   └── utils/                 # Utility modules
│       ├── video_processor.py # Video I/O operations
│       └── visualizer.py      # Result visualization
├── dataset/                   # Training and validation data
├── models/                    # Trained model weights
├── configs/                   # Configuration files
└── notebooks/                 # Additional notebooks
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Khaled100233/Robot-Referee.git
cd Robot-Referee

# Install dependencies
pip install -r requirements.txt
```

### Local Inference

```bash
# Run inference on a video
python main.py --video path/to/video.mp4 --output path/to/output.mp4

# Use custom models
python main.py \
  --video input.mp4 \
  --output output.mp4 \
  --pose-model models/custom_pose.pt \
  --detect-model models/custom_detect.pt
```

### Training on Google Colab

1. Upload `train.ipynb` to Google Colab
2. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
3. Upload your dataset or mount Google Drive
4. Update `dataset/data.yaml` with your dataset paths
5. Run all cells to start training

**Cost Optimization Features:**
- ✅ Early Stopping (saves compute when model stops improving)
- ✅ Mixed Precision Training (AMP for 1.5-2x speedup)
- ✅ Efficient batch sizing
- ✅ Model checkpointing for resume capability

## 📊 Dataset Structure

Your dataset should follow this structure:

```
dataset/
├── data.yaml              # Dataset configuration
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images (optional)
└── labels/
    ├── train/            # Training labels (YOLO format)
    ├── val/              # Validation labels
    └── test/             # Test labels (optional)
```

### data.yaml Example

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: player
  1: ball
  2: referee

kpt_shape: [17, 3]  # For pose detection
```

## 🛠️ Configuration

Edit `config.py` to customize:

- Model paths and types
- Detection thresholds
- Case-specific parameters
- Training hyperparameters
- Video processing settings

### Key Configuration Parameters

```python
# Unnatural Arm Detection
CASES_CONFIG["unnatural_arm"]["angle_threshold"] = 135  # degrees

# Reaction Time
CASES_CONFIG["reaction_time"]["max_time_ms"] = 500  # milliseconds

# Deflection Detection
CASES_CONFIG["deflection"]["min_trajectory_change"] = 15  # degrees
```

## 🧪 Development Workflow

### File Headers

All Python files include standardized headers for AI/Human readability:

```python
"""
STATUS: Active/Development/Deprecated
PURPOSE: Brief description of file purpose
NEXT_STEPS:
  - Future improvements
  - Known limitations
  - Planned features
"""
```

### Best Practices

1. **Local Development**: Use `main.py` for inference and testing
2. **Heavy Training**: Use `train.ipynb` on Colab with GPU
3. **Cost Optimization**: Always enable early stopping and mixed precision
4. **Version Control**: Track model versions in `models/` directory

## 📈 Performance Metrics

Monitor these metrics during training:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: mAP across IoU thresholds 0.5 to 0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## 🎯 Use Cases

- Real-time referee assistance systems
- Post-match video analysis
- Training and educational tools
- Automated highlight generation
- Sports analytics platforms

## 🔧 Technical Stack

- **Python 3.8+**
- **Ultralytics YOLOv8** - Pose estimation and object detection
- **OpenCV** - Video processing and visualization
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **Google Colab** - Cloud training platform

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with proper headers
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

**Project Maintainer:** Khaled100233  
**Repository:** [Robot-Referee](https://github.com/Khaled100233/Robot-Referee)

## 🙏 Acknowledgments

- Ultralytics team for YOLOv8
- OpenCV community
- Football rules and handball guidelines from FIFA

---

**⚠️ Important Notes:**

1. This system is designed to assist referees, not replace them
2. Model accuracy depends on training data quality
3. Always validate detections with human review
4. Follow football governing body guidelines for video assistant systems

**🎓 Research & Education:**

This project is suitable for:
- Computer Vision research
- Sports analytics studies
- AI ethics discussions (automated decision-making in sports)
- Educational purposes in ML/DL courses

---

Made with ⚽ and 🤖 by the Robot Referee team
