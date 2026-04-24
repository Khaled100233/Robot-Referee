# 🤖 Robot Referee — AI-Powered Handball Foul Detection

> Automatically detect handball fouls in football (soccer) videos using computer vision and pose estimation, grounded in FIFA's official Law 12 rules.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

Handball decisions in football are among the most controversial calls in the sport. Referees must make split-second judgements about a player's arm position, reaction time, and intention — all of which are highly subjective.

**Robot Referee** replaces guesswork with objective, measurable data. It processes football video clips and analyses each frame for potential handball infractions across **three independent FIFA Law 12 criteria**:

| Case | Criterion | Detection Method |
|------|-----------|-----------------|
| 1 | Arm making the body "unnaturally bigger" | Arm-torso angle + ball proximity |
| 2 | Player had time to react | Ball speed → reaction time calculation |
| 3 | Deliberate movement of the arm toward the ball | Arm velocity vector vs ball direction |

---

## ✨ Features

- 🦴 **Pose estimation** — YOLOv8-pose detects 17 body keypoints (shoulders, elbows, wrists, hips) per player per frame
- ⚽ **Ball tracking** — YOLOv8 ball detection with configurable confidence thresholds
- 📐 **Case 1:** Measures the arm-torso angle and flags positions that make the body unnaturally bigger
- ⏱️ **Case 2:** Computes ball speed and calculates whether the player had enough time to react (< 200 ms → no foul; > 400 ms → possible foul)
- 🎯 **Case 3:** Tracks wrist/elbow velocity vectors to detect deliberate movement toward the ball, confirmed by ball-trajectory deflection
- 🎬 **Annotated output video** — bounding boxes, skeleton overlays, angle labels, reaction-time readings, and verdict overlays
- 🐍 **Clean Python API** — single `analyze_handball()` call returns a structured result dict

---

## 🏗️ Architecture

```
Robot-Referee/
├── main.py               # Quick-start vision pipeline (pose + ball detection)
├── handball_foul.py      # Core analysis module — all 3 FIFA Law 12 cases
├── requirements.txt      # Python dependencies
├── input.mp4             # Sample input video
├── output/               # Annotated output videos
├── datasets/             # Training data (YOLOv8 format)
└── notebooks/
    ├── 01_data_pre_processing.ipynb   # Dataset preparation
    ├── case1.ipynb                    # Case 1 prototype & explanation
    ├── case2.ipynb                    # Case 2 prototype & explanation
    ├── case3.ipynb                    # Case 3 prototype & explanation
    └── use.ipynb                      # End-to-end usage demo
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/Khaled100233/Robot-Referee.git
cd Robot-Referee

# 2. Install dependencies
pip install -r requirements.txt
```

The first run will automatically download the pretrained `yolov8n-pose.pt` and `yolov8x.pt` weights from Ultralytics.

---

## 🚀 Usage

### Python API

```python
from handball_foul import analyze_handball

result = analyze_handball(
    video_path="input.mp4",
    output_path="output/annotated.mp4",   # optional — omit to skip video writing
    verbose=True,
)

print(result["final_verdict"])   # e.g. "HANDBALL DETECTED"
print(result["is_foul"])         # True / False
print(result["reason"])          # Human-readable explanation
```

#### Return value keys

| Key | Type | Description |
|-----|------|-------------|
| `final_verdict` | `str` | Overall verdict string |
| `is_foul` | `bool` | `True` if a handball foul is detected |
| `reason` | `str` | Human-readable explanation of the verdict |
| `case1` | `dict` | Case 1 detailed results (arm angle analysis) |
| `case2` | `dict` | Case 2 detailed results (reaction time) |
| `case3` | `dict` | Case 3 detailed results (deliberate movement) |
| `total_frames` | `int` | Total frames processed |
| `fps` | `int` | Video frame rate |

### Command Line

```bash
python handball_foul.py input.mp4
```

### Quick Vision Test (pose + ball only, no foul logic)

```bash
python main.py
```

Output is saved to `output/vision_test.mp4`.

---

## 🔬 Case Details

### Case 1 — Arm "Unnaturally Bigger" (FIFA Law 12)

The arm-torso angle (shoulder → hip vs shoulder → elbow) is measured for every player in every frame. Risk is classified as:

- **LOW** (< 30°) — arm close to body, no foul
- **MEDIUM** (30°–70°) — possible foul if ball is nearby
- **HIGH** (> 70°) — arm unnaturally extended; handball if ball is within 150 px

### Case 2 — Reaction Time

Ball speed is calculated from frame-to-frame position history. Combined with the distance from ball to arm, the time available for the player to react is computed:

```
reaction_time_ms = (distance_px / ball_speed_px_s) × 1000
```

| Reaction Time | Verdict |
|--------------|---------|
| < 200 ms | No Foul — insufficient time to react |
| 200–400 ms | Debatable — borderline |
| > 400 ms | Possible Foul — player had time to move |

### Case 3 — Deliberate Movement

Wrist and elbow velocity vectors (averaged over the last 5 frames) are compared with the arm→ball direction using cosine similarity. A ball trajectory change after proximity confirms contact:

| Arm Toward Ball? | Ball Deflected? | Verdict |
|-----------------|----------------|---------|
| No | — | No Foul |
| Yes | No | Possible Handball |
| Yes | Yes | **Deliberate Handball Detected** |

---

## 📓 Notebooks

The `notebooks/` directory contains step-by-step Jupyter notebooks:

- **`01_data_pre_processing.ipynb`** — prepare and augment the training dataset
- **`case1.ipynb`** — explore and prototype the arm-angle detection algorithm
- **`case2.ipynb`** — explore and prototype the reaction-time algorithm
- **`case3.ipynb`** — explore and prototype the deliberate-movement algorithm
- **`use.ipynb`** — end-to-end demonstration of the full pipeline

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for pose estimation and object detection
- [IFAB Laws of the Game — Law 12](https://www.theifab.com/laws/latest/fouls-and-misconduct/) for the official handball criteria
- [Roboflow](https://roboflow.com) for dataset management tooling
