# Quick Start Guide - Robot Referee

Get up and running with Robot Referee in 5 minutes!

## 🚀 Step 1: Installation (1 min)

```bash
# Clone repository
git clone https://github.com/Khaled100233/Robot-Referee.git
cd Robot-Referee

# Install dependencies
pip install -r requirements.txt
```

## 📁 Step 2: Prepare Dataset (Optional for inference)

If you want to **train** your own models:

```bash
# Create dataset structure
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}

# Add your images and labels
# Labels should be in YOLO format (one .txt per image)
```

Edit `dataset/data.yaml`:
```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: player
  1: ball

kpt_shape: [17, 3]
```

## 🎯 Step 3: Run Inference (1 min)

### Option A: Use Pre-trained Models
Models will auto-download on first use:

```bash
python main.py --video your_video.mp4 --output result.mp4
```

### Option B: Use Custom Models
```bash
python main.py \
  --video your_video.mp4 \
  --output result.mp4 \
  --pose-model models/custom_pose.pt
```

## 🔧 Step 4: Configure (Optional)

Edit `config.py` to adjust thresholds:

```python
CASES_CONFIG = {
    "unnatural_arm": {
        "enabled": True,
        "angle_threshold": 135,  # Adjust this
        "min_confidence": 0.6
    },
    "reaction_time": {
        "enabled": True,
        "max_time_ms": 500  # Adjust this
    },
    "deflection": {
        "enabled": True,
        "min_trajectory_change": 15  # Adjust this
    }
}
```

## 🎓 Step 5: Train on Colab (For custom models)

1. Open Google Colab: https://colab.research.google.com/
2. Upload `train.ipynb`
3. Change runtime to GPU: Runtime → Change runtime type → GPU
4. Mount Google Drive (optional for dataset storage)
5. Upload dataset or link to Drive
6. Run all cells
7. Download trained model

### Training Configuration

In the notebook, adjust:
```python
CONFIG = {
    "model_type": "yolov8n-pose",  # Options: n, s, m, l
    "epochs": 100,
    "batch_size": 16,
    "patience": 10,      # Early stopping
    "amp": True,         # Mixed precision
}
```

## 📊 Step 6: Review Results

After inference, check:
- Output video with annotations
- Console output for detection summary
- Per-case detection counts

Example output:
```
==================================================
DETECTION SUMMARY
==================================================
UNNATURAL_ARM: 5 detections
REACTION_TIME: 2 detections
DEFLECTION: 3 detections
==================================================
```

## 🔍 Troubleshooting

### Issue: "Cannot open video"
**Solution:** Check video path and codec support
```bash
# Test with ffmpeg
ffmpeg -i your_video.mp4
```

### Issue: "No module named 'ultralytics'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use smaller model
```python
# In config.py
TRAINING_CONFIG["batch_size"] = 8  # Reduce from 16
```

### Issue: Models downloading slowly
**Solution:** Pre-download models
```bash
# In Python
from ultralytics import YOLO
YOLO("yolov8n-pose.pt")  # Downloads to cache
YOLO("yolov8n.pt")
```

## 💡 Pro Tips

1. **Start Small**: Use `yolov8n` for quick testing
2. **GPU Required**: Training needs GPU (use Colab free tier)
3. **Video Quality**: Higher resolution = better detection
4. **Frame Rate**: 30 FPS minimum recommended
5. **Lighting**: Consistent lighting improves accuracy

## 📈 Expected Results

### First Run (Pre-trained)
- Detection rate: 40-60%
- False positives: ~30%
- Processing speed: 20-30 FPS

### After Training (Custom data)
- Detection rate: 70-85%
- False positives: ~10%
- Processing speed: 20-30 FPS

## 🎯 Next Actions

1. **Test with sample video**: Verify setup works
2. **Collect training data**: Annotate 500+ images
3. **Train custom model**: Use your dataset
4. **Fine-tune thresholds**: Adjust based on results
5. **Deploy**: Integrate into your workflow

## 📚 Additional Resources

- **Full Documentation**: See README.md
- **Training Guide**: See train.ipynb comments
- **Dataset Format**: See dataset/README.md
- **API Reference**: See docstrings in source code

## 🆘 Getting Help

1. Check documentation in README.md
2. Review PROJECT_SUMMARY.md for architecture
3. Check file headers for specific modules
4. Review GitHub issues
5. Check Ultralytics YOLO docs: https://docs.ultralytics.com/

## ⚡ Quick Commands Cheat Sheet

```bash
# Basic inference
python main.py --video input.mp4 --output output.mp4

# Check config
python config.py

# Test imports
python -c "from src.cases.unnatural_arm import UnnaturalArmDetector"

# Compile all files
python -m py_compile main.py config.py src/**/*.py

# Install dev dependencies
pip install jupyter ipython ipdb

# Start Jupyter for experiments
jupyter notebook notebooks/
```

---

**Ready to go!** 🚀

For detailed information, see README.md
