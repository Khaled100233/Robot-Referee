import cv2
from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
VIDEO_PATH = "input.mp4"         # Upload a short football clip and name it this
OUTPUT_PATH = "output/vision_test.mp4"
MODEL_POSE = "yolov8n-pose.pt"   # Pre-trained pose model
MODEL_BALL = "models/best.pt"    # YOUR custom trained model

# ---------------------------------------------------------
# MAIN PROCESSING LOOP
# ---------------------------------------------------------
def run_vision_system():
    # 1. Load both models
    print("Loading AI models...")
    pose_model = YOLO(MODEL_POSE)
    ball_model = YOLO(MODEL_BALL)

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create Video Writer
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps}fps")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 10 == 0: print(f"Processing frame {frame_count}...")

        # -------------------------------------------------
        # AI LAYER
        # -------------------------------------------------
        
        # A. Run Pose Detection (Players)
        # conf=0.5 means "Only show if 50% sure it's a person"
        pose_results = pose_model(frame, verbose=False, conf=0.5)
        
        # B. Run Ball Detection (Custom)
        # classes=[0] ensures we only look for the 'ball' class from your model
        ball_results = ball_model(frame, verbose=False, conf=0.5)

        # -------------------------------------------------
        # VISUALIZATION LAYER
        # -------------------------------------------------
        
        # Draw the Skeletons (Pose)
        annotated_frame = pose_results[0].plot()
        
        # Draw the Ball Box (Custom) - Overlay on top
        # We manually draw this to ensure it's distinct
        for box in ball_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw a thick ORANGE box for the ball
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(annotated_frame, "BALL", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Save frame
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"✅ Success! Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_vision_system()
    print("Done processing video. Check the output folder for results.")