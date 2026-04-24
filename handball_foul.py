"""
Handball Foul Detection Module
===============================
Combines three FIFA Law 12 handball evaluation cases into a single analysis:

  Case 1 – Arm making the body "unnaturally bigger"
           (arm-torso angle + ball proximity)

  Case 2 – Reaction time and distance
           (ball speed + time available to react)

  Case 3 – Deliberate movement of the arm toward the ball
           (arm velocity cosine similarity + ball deflection)

Usage:
    from handball_foul import analyze_handball
    result = analyze_handball("input.mp4")
    print(result["final_verdict"])
    print(result["reason"])

Or from the command line:
    python handball_foul.py input.mp4
"""

import cv2
import numpy as np
import math
from collections import deque
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model paths (relative to project root) ---
DEFAULT_MODEL_POSE = "yolov8n-pose.pt"
DEFAULT_MODEL_BALL = "yolov8x.pt"

# --- COCO keypoint indices ---
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

# --- Case 1: Arm angle thresholds (degrees) ---
ANGLE_LOW = 30       # Below → LOW risk (arm close to body)
ANGLE_HIGH = 70      # Above → HIGH risk (arm "unnaturally bigger")

# --- Common: Ball proximity threshold (pixels) ---
BALL_DISTANCE_THRESHOLD = 150

# --- Case 2: Reaction time thresholds (ms) ---
REACTION_TIME_NO_FOUL = 200    # Below → player cannot react
REACTION_TIME_DEBATABLE = 400  # Above → player had time to react
BALL_HISTORY_LENGTH = 5        # Frames for ball speed smoothing

# --- Case 3: Deliberate movement thresholds ---
ARM_TOWARD_BALL_THRESHOLD = 0.5   # Cosine similarity threshold
BALL_DIRECTION_CHANGE_THRESHOLD = 30  # Degrees for ball deflection
HISTORY_LENGTH = 5                 # Frames for motion vector history
POST_CONTACT_WINDOW = 5           # Frames after contact to check deflection

# --- Ball class ID (COCO: 32 = sports ball; custom model: [0]) ---
BALL_CLASS_ID = [32]


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def is_keypoint_valid(kp):
    """Check if a keypoint was actually detected (not at origin 0,0)."""
    return kp[0] != 0 or kp[1] != 0


def get_ball_center(ball_results):
    """
    Extract the ball center from detection results.
    Returns (cx, cy) of the highest-confidence detection, or None.
    """
    if len(ball_results[0].boxes) > 0:
        best_box = None
        best_conf = 0
        for box in ball_results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box
        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    return None


def get_all_ball_centers(ball_results):
    """Extract all ball center points as a list of (cx, cy) tuples."""
    centers = []
    if len(ball_results[0].boxes) > 0:
        for box in ball_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return centers


def min_distance_to_ball(arm_keypoints, ball_centers):
    """
    Minimum Euclidean distance between any arm keypoint and any ball center.
    Returns (min_dist, closest_arm_point, closest_ball_point) or (None, None, None).
    """
    min_dist = float('inf')
    closest_arm = None
    closest_ball = None

    for kp in arm_keypoints:
        if not is_keypoint_valid(kp):
            continue
        for bc in ball_centers:
            dist = math.sqrt((kp[0] - bc[0]) ** 2 + (kp[1] - bc[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_arm = kp
                closest_ball = bc

    if min_dist == float('inf'):
        return None, None, None
    return min_dist, closest_arm, closest_ball


# =============================================================================
# CASE 1 HELPERS — Arm Angle (Unnaturally Bigger)
# =============================================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Angle at point_a between vectors (point_a → point_b) and (point_a → point_c).
    Used as: shoulder → hip (torso) vs shoulder → elbow (arm).
    Returns angle in degrees, or None if a vector is zero-length.
    """
    vec_torso = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    vec_arm = np.array([point_c[0] - point_a[0], point_c[1] - point_a[1]])

    mag_torso = np.linalg.norm(vec_torso)
    mag_arm = np.linalg.norm(vec_arm)

    if mag_torso == 0 or mag_arm == 0:
        return None

    cos_angle = np.clip(np.dot(vec_torso, vec_arm) / (mag_torso * mag_arm), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def classify_risk(angle):
    """Classify arm-torso angle risk: LOW / MEDIUM / HIGH / UNKNOWN."""
    if angle is None:
        return "UNKNOWN"
    if angle < ANGLE_LOW:
        return "LOW"
    elif angle <= ANGLE_HIGH:
        return "MEDIUM"
    else:
        return "HIGH"


def make_handball_decision_case1(arm_risk, ball_distance):
    """
    Case 1 decision: combine arm angle risk + ball proximity.
    Returns (decision_string, severity_int).
    severity: 0=no foul, 1=possible, 2=handball detected
    """
    if ball_distance is None:
        return "NO HANDBALL", 0

    ball_close = ball_distance < BALL_DISTANCE_THRESHOLD

    if arm_risk == "HIGH" and ball_close:
        return "HANDBALL DETECTED", 2
    elif arm_risk == "MEDIUM" and ball_close:
        return "POSSIBLE HANDBALL", 1
    else:
        return "NO HANDBALL", 0


# =============================================================================
# CASE 2 HELPERS — Reaction Time
# =============================================================================

def calculate_ball_speed(ball_history, fps):
    """
    Average ball speed from position history.
    Returns (speed_px_per_sec, speed_px_per_frame) or (None, None).
    """
    if len(ball_history) < 2:
        return None, None

    displacements = []
    for i in range(1, len(ball_history)):
        dx = ball_history[i][0] - ball_history[i - 1][0]
        dy = ball_history[i][1] - ball_history[i - 1][1]
        displacements.append(math.sqrt(dx ** 2 + dy ** 2))

    speed_px_per_frame = sum(displacements) / len(displacements)
    return speed_px_per_frame * fps, speed_px_per_frame


def calculate_reaction_time_ms(distance_px, ball_speed_px_per_sec):
    """
    Reaction time in milliseconds: time the ball took to cover distance_px.
    Returns None if speed is unknown or essentially zero.
    """
    if ball_speed_px_per_sec is None or ball_speed_px_per_sec < 1.0:
        return None
    return (distance_px / ball_speed_px_per_sec) * 1000


def classify_reaction_time(reaction_time_ms):
    """Classify reaction time into NO FOUL / DEBATABLE / POSSIBLE FOUL / UNKNOWN."""
    if reaction_time_ms is None:
        return "UNKNOWN"
    if reaction_time_ms < REACTION_TIME_NO_FOUL:
        return "NO FOUL"
    elif reaction_time_ms <= REACTION_TIME_DEBATABLE:
        return "DEBATABLE"
    else:
        return "POSSIBLE FOUL"


# =============================================================================
# CASE 3 HELPERS — Deliberate Arm Movement
# =============================================================================

def compute_velocity_vector(history):
    """
    Average velocity vector from a position history (deque/list of (x, y)).
    Returns (vx, vy) or (None, None) if insufficient data.
    """
    if len(history) < 2:
        return None, None

    total_dx = total_dy = 0
    count = 0
    for i in range(1, len(history)):
        total_dx += history[i][0] - history[i - 1][0]
        total_dy += history[i][1] - history[i - 1][1]
        count += 1
    return total_dx / count, total_dy / count


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two 2D vectors. Returns value in [-1, 1] or None."""
    mag_a = math.sqrt(vec_a[0] ** 2 + vec_a[1] ** 2)
    mag_b = math.sqrt(vec_b[0] ** 2 + vec_b[1] ** 2)
    if mag_a < 1e-6 or mag_b < 1e-6:
        return None
    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    return max(-1.0, min(1.0, dot / (mag_a * mag_b)))


def angle_between_vectors(vec_a, vec_b):
    """Angle in degrees [0, 180] between two 2D vectors, or None."""
    cs = cosine_similarity(vec_a, vec_b)
    if cs is None:
        return None
    return math.degrees(math.acos(cs))


def compute_ball_direction_change(ball_history, contact_index, window=POST_CONTACT_WINDOW):
    """
    Angular change of ball direction before vs after a contact event.
    ball_history: list of (frame_num, cx, cy).
    Returns degrees or None.
    """
    if contact_index < 1 or contact_index >= len(ball_history) - 1:
        return None

    before_start = max(0, contact_index - window)
    before_positions = ball_history[before_start:contact_index + 1]
    if len(before_positions) < 2:
        return None
    bvx, bvy = compute_velocity_vector([(p[1], p[2]) for p in before_positions])
    if bvx is None:
        return None

    after_end = min(len(ball_history), contact_index + 1 + window)
    after_positions = ball_history[contact_index:after_end]
    if len(after_positions) < 2:
        return None
    avx, avy = compute_velocity_vector([(p[1], p[2]) for p in after_positions])
    if avx is None:
        return None

    return angle_between_vectors((bvx, bvy), (avx, avy))


def classify_arm_movement(cos_sim):
    """Classify arm movement direction relative to ball."""
    if cos_sim is None:
        return "UNKNOWN"
    if cos_sim > ARM_TOWARD_BALL_THRESHOLD:
        return "TOWARD BALL"
    elif cos_sim < -ARM_TOWARD_BALL_THRESHOLD:
        return "AWAY FROM BALL"
    else:
        return "NEUTRAL"


def make_deliberate_decision(arm_movement, ball_deflected):
    """
    Case 3 decision: combine arm movement direction + ball deflection.
    Returns (decision_string, severity_int).
    severity: 0=no foul, 1=unlikely/possible, 2=deliberate
    """
    if arm_movement == "TOWARD BALL":
        if ball_deflected:
            return "DELIBERATE HANDBALL", 2
        else:
            return "POSSIBLE HANDBALL", 1
    elif arm_movement == "NEUTRAL":
        if ball_deflected:
            return "UNLIKELY FOUL", 1
        else:
            return "NO FOUL", 0
    else:
        return "NO FOUL", 0


# =============================================================================
# MAIN ANALYSIS — SINGLE-PASS OVER VIDEO
# =============================================================================

def analyze_handball(
    video_path,
    model_pose_path=None,
    model_ball_path=None,
    output_path=None,
    verbose=True,
):
    """
    Analyse a football video for handball fouls using all three FIFA Law 12 cases.

    Args:
        video_path:       Path to the input video file.
        model_pose_path:  Path to YOLOv8-pose model (default: yolov8n-pose.pt).
        model_ball_path:  Path to YOLOv8 ball detection model (default: yolov8x.pt).
        output_path:      Optional path to save annotated output video.
                          If None, no output video is written.
        verbose:          Print progress to stdout.

    Returns:
        dict with keys:
            final_verdict   – str:  Overall verdict string.
            is_foul         – bool: True if handball foul is detected.
            reason          – str:  Human-readable explanation.
            case1           – dict: Case 1 detailed results.
            case2           – dict: Case 2 detailed results.
            case3           – dict: Case 3 detailed results.
            total_frames    – int:  Number of frames processed.
            fps             – int:  Video FPS.
    """

    if model_pose_path is None:
        model_pose_path = DEFAULT_MODEL_POSE
    if model_ball_path is None:
        model_ball_path = DEFAULT_MODEL_BALL

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    if verbose:
        print("Loading AI models...")
    pose_model = YOLO(model_pose_path)
    ball_model = YOLO(model_ball_path)
    if verbose:
        print("Models loaded.\n")

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if verbose:
        print(f"Video: {width}x{height} @ {fps}fps — {total_frames} frames")
        print("-" * 60)

    # ------------------------------------------------------------------
    # Tracking state
    # ------------------------------------------------------------------
    # Case 2 — ball speed
    ball_speed_history = deque(maxlen=BALL_HISTORY_LENGTH)

    # Case 3 — motion vectors
    ball_history_full = []                       # (frame_num, cx, cy)
    ball_history_recent = deque(maxlen=HISTORY_LENGTH)
    global_arm_histories = {
        kp_idx: deque(maxlen=HISTORY_LENGTH)
        for kp_idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW]
    }
    pending_contacts = []

    # ------------------------------------------------------------------
    # Result accumulators
    # ------------------------------------------------------------------
    # Case 1
    c1_handball_frames = []
    c1_possible_frames = []

    # Case 2
    c2_no_foul_frames = []
    c2_debatable_frames = []
    c2_possible_foul_frames = []
    c2_reaction_times = []          # (frame_num, rt_ms, verdict)

    # Case 3
    c3_deliberate_frames = []
    c3_possible_frames = []
    c3_unlikely_frames = []
    c3_contact_events = []

    frame_count = 0

    # ------------------------------------------------------------------
    # Frame-by-frame processing
    # ------------------------------------------------------------------
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        if verbose and frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames}...")

        # ---- AI inference ----
        pose_results = pose_model(frame, verbose=False, conf=0.5)
        ball_results = ball_model(frame, verbose=False, conf=0.3, classes=BALL_CLASS_ID)

        # ---- Ball tracking ----
        ball_center = get_ball_center(ball_results)
        if ball_center is not None:
            ball_speed_history.append(ball_center)
            ball_history_full.append((frame_count, ball_center[0], ball_center[1]))
            ball_history_recent.append(ball_center)

        ball_centers = get_all_ball_centers(ball_results)
        ball_speed_px_s, _ = calculate_ball_speed(ball_speed_history, fps)

        # ---- Track closest person's arms for Case 3 ----
        if (pose_results[0].keypoints is not None
                and len(pose_results[0].keypoints) > 0
                and ball_center is not None):
            keypoints_all = pose_results[0].keypoints.xy.cpu().numpy()
            best_person = None
            best_dist = float('inf')
            for pidx, kpts in enumerate(keypoints_all):
                for kp_idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW]:
                    kp = kpts[kp_idx]
                    if is_keypoint_valid(kp):
                        d = math.sqrt((kp[0] - ball_center[0]) ** 2 +
                                      (kp[1] - ball_center[1]) ** 2)
                        if d < best_dist:
                            best_dist = d
                            best_person = pidx
            if best_person is not None:
                kpts = keypoints_all[best_person]
                for kp_idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW]:
                    kp = kpts[kp_idx]
                    if is_keypoint_valid(kp):
                        global_arm_histories[kp_idx].append((float(kp[0]), float(kp[1])))

        # ---- Prepare annotated frame (only if writing output) ----
        annotated_frame = None
        if out is not None:
            annotated_frame = pose_results[0].plot()
            for box in ball_results[0].boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
                lbl = f"BALL | {ball_speed_px_s:.0f} px/s" if ball_speed_px_s else "BALL"
                cv2.putText(annotated_frame, lbl, (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # ---- Per-frame decisions ----
        frame_c1_decision = "NO HANDBALL"
        frame_c2_verdict = "NO FOUL"
        frame_c3_decision = "NO FOUL"

        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
            keypoints_all = pose_results[0].keypoints.xy.cpu().numpy()

            for person_idx, kpts in enumerate(keypoints_all):

                # ==============================================================
                # CASE 1 — Arm angle analysis (left + right)
                # ==============================================================
                for side in ["left", "right"]:
                    if side == "left":
                        shoulder, elbow, wrist, hip = (
                            kpts[LEFT_SHOULDER], kpts[LEFT_ELBOW],
                            kpts[LEFT_WRIST], kpts[LEFT_HIP],
                        )
                    else:
                        shoulder, elbow, wrist, hip = (
                            kpts[RIGHT_SHOULDER], kpts[RIGHT_ELBOW],
                            kpts[RIGHT_WRIST], kpts[RIGHT_HIP],
                        )

                    if all(is_keypoint_valid(p) for p in [shoulder, elbow, hip]):
                        angle = calculate_angle(shoulder, hip, elbow)
                        risk = classify_risk(angle)

                        arm_kps = [k for k in [elbow, wrist] if is_keypoint_valid(k)]
                        dist, arm_pt, ball_pt = min_distance_to_ball(arm_kps, ball_centers)
                        decision, severity = make_handball_decision_case1(risk, dist)

                        if severity == 2 and frame_c1_decision != "HANDBALL DETECTED":
                            frame_c1_decision = "HANDBALL DETECTED"
                        elif severity == 1 and frame_c1_decision == "NO HANDBALL":
                            frame_c1_decision = "POSSIBLE HANDBALL"

                        # annotate
                        if annotated_frame is not None and angle is not None:
                            color = {
                                "LOW": (0, 255, 0), "MEDIUM": (0, 165, 255),
                                "HIGH": (0, 0, 255)
                            }.get(risk, (200, 200, 200))
                            prefix = "L" if side == "left" else "R"
                            cv2.putText(
                                annotated_frame,
                                f"{prefix}:{angle:.0f} {risk}",
                                (int(shoulder[0]) + (-70 if side == "left" else 10),
                                 int(shoulder[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
                            )

                # ==============================================================
                # CASE 2 — Reaction time
                # ==============================================================
                arm_kps = [
                    kpts[idx]
                    for idx in [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
                    if is_keypoint_valid(kpts[idx])
                ]
                if arm_kps:
                    dist, arm_pt, ball_pt = min_distance_to_ball(arm_kps, ball_centers)
                    if dist is not None and dist < BALL_DISTANCE_THRESHOLD:
                        rt = calculate_reaction_time_ms(dist, ball_speed_px_s)
                        verdict = classify_reaction_time(rt)
                        c2_reaction_times.append((frame_count, rt, verdict))

                        if verdict == "POSSIBLE FOUL":
                            frame_c2_verdict = "POSSIBLE FOUL"
                        elif verdict == "DEBATABLE" and frame_c2_verdict != "POSSIBLE FOUL":
                            frame_c2_verdict = "DEBATABLE"

                        # annotate
                        if annotated_frame is not None and arm_pt is not None and ball_pt is not None:
                            vc = {"NO FOUL": (0, 255, 0), "DEBATABLE": (0, 165, 255),
                                  "POSSIBLE FOUL": (0, 0, 255)}.get(verdict, (200, 200, 200))
                            p1 = (int(arm_pt[0]), int(arm_pt[1]))
                            p2 = (int(ball_pt[0]), int(ball_pt[1]))
                            cv2.line(annotated_frame, p1, p2, vc, 2)
                            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                            rt_s = f"{rt:.0f}ms" if rt is not None else "N/A"
                            cv2.putText(annotated_frame, f"{dist:.0f}px|{rt_s}|{verdict}",
                                        (mid[0] - 60, mid[1] - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, vc, 1)

                # ==============================================================
                # CASE 3 — Deliberate arm movement
                # ==============================================================
                for kp_idx, kp_name in [
                    (LEFT_WRIST, "LW"), (RIGHT_WRIST, "RW"),
                    (LEFT_ELBOW, "LE"), (RIGHT_ELBOW, "RE"),
                ]:
                    kp = kpts[kp_idx]
                    if not is_keypoint_valid(kp):
                        continue

                    history = global_arm_histories.get(kp_idx)
                    if history is None or len(history) < 2:
                        continue

                    avx, avy = compute_velocity_vector(history)
                    if avx is None:
                        continue
                    arm_speed = math.sqrt(avx ** 2 + avy ** 2)
                    if arm_speed < 0.5:
                        continue

                    cos_sim_val = None
                    dist_to_ball = None
                    closest_ball_pt = None
                    if ball_centers:
                        d, _, bp = min_distance_to_ball([kp], ball_centers)
                        if d is not None:
                            dist_to_ball = d
                            closest_ball_pt = bp
                            arm_to_ball = (bp[0] - kp[0], bp[1] - kp[1])
                            cos_sim_val = cosine_similarity((avx, avy), arm_to_ball)

                    movement = classify_arm_movement(cos_sim_val)

                    # Contact event?
                    if dist_to_ball is not None and dist_to_ball < BALL_DISTANCE_THRESHOLD:
                        ball_deflected = False
                        dir_change = None
                        if len(ball_history_full) >= 3:
                            contact_idx = len(ball_history_full) - 1
                            dir_change = compute_ball_direction_change(
                                ball_history_full, contact_idx, POST_CONTACT_WINDOW
                            )
                            if dir_change is not None and dir_change > BALL_DIRECTION_CHANGE_THRESHOLD:
                                ball_deflected = True

                        pending_contacts.append({
                            'frame': frame_count,
                            'ball_hist_idx': len(ball_history_full) - 1,
                            'movement': movement,
                        })

                        decision, severity = make_deliberate_decision(movement, ball_deflected)
                        c3_contact_events.append({
                            'frame': frame_count,
                            'keypoint': kp_name,
                            'cos_sim': cos_sim_val,
                            'movement': movement,
                            'ball_dist': dist_to_ball,
                            'ball_deflected': ball_deflected,
                            'direction_change': dir_change,
                            'decision': decision,
                        })

                        if decision == "DELIBERATE HANDBALL":
                            frame_c3_decision = "DELIBERATE HANDBALL"
                        elif decision == "POSSIBLE HANDBALL" and frame_c3_decision != "DELIBERATE HANDBALL":
                            frame_c3_decision = "POSSIBLE HANDBALL"
                        elif decision == "UNLIKELY FOUL" and frame_c3_decision not in [
                            "DELIBERATE HANDBALL", "POSSIBLE HANDBALL"
                        ]:
                            frame_c3_decision = "UNLIKELY FOUL"

                        # annotate
                        if annotated_frame is not None and closest_ball_pt is not None:
                            mc = {"TOWARD BALL": (0, 0, 255), "NEUTRAL": (0, 165, 255),
                                  "AWAY FROM BALL": (0, 255, 0)}.get(movement, (200, 200, 200))
                            ap = (int(kp[0]), int(kp[1]))
                            arrow_end = (int(kp[0] + avx * 8), int(kp[1] + avy * 8))
                            cv2.arrowedLine(annotated_frame, ap, arrow_end, mc, 2, tipLength=0.3)

        # ---- Accumulate frame-level results ----
        if frame_c1_decision == "HANDBALL DETECTED":
            c1_handball_frames.append(frame_count)
        elif frame_c1_decision == "POSSIBLE HANDBALL":
            c1_possible_frames.append(frame_count)

        if frame_c2_verdict == "POSSIBLE FOUL":
            c2_possible_foul_frames.append(frame_count)
        elif frame_c2_verdict == "DEBATABLE":
            c2_debatable_frames.append(frame_count)
        elif frame_c2_verdict == "NO FOUL" and any(
            rt[2] == "NO FOUL" for rt in c2_reaction_times if rt[0] == frame_count
        ):
            c2_no_foul_frames.append(frame_count)

        if frame_c3_decision == "DELIBERATE HANDBALL":
            c3_deliberate_frames.append(frame_count)
        elif frame_c3_decision == "POSSIBLE HANDBALL":
            c3_possible_frames.append(frame_count)
        elif frame_c3_decision == "UNLIKELY FOUL":
            c3_unlikely_frames.append(frame_count)

        # ---- Draw combined banner on annotated frame ----
        if annotated_frame is not None:
            _draw_combined_banner(
                annotated_frame, frame_c1_decision, frame_c2_verdict, frame_c3_decision
            )
            out.write(annotated_frame)

    # ------------------------------------------------------------------
    # Post-processing: re-evaluate Case 3 pending contacts
    # ------------------------------------------------------------------
    for pc in pending_contacts:
        idx = pc['ball_hist_idx']
        if idx + POST_CONTACT_WINDOW < len(ball_history_full):
            dc = compute_ball_direction_change(ball_history_full, idx, POST_CONTACT_WINDOW)
            if dc is not None and dc > BALL_DIRECTION_CHANGE_THRESHOLD:
                if pc['movement'] == "TOWARD BALL" and pc['frame'] not in c3_deliberate_frames:
                    c3_deliberate_frames.append(pc['frame'])
                    if pc['frame'] in c3_possible_frames:
                        c3_possible_frames.remove(pc['frame'])

    c3_deliberate_frames.sort()
    c3_possible_frames.sort()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cap.release()
    if out is not None:
        out.release()

    # ------------------------------------------------------------------
    # Build per-case results
    # ------------------------------------------------------------------
    case1 = {
        'handball_frames': c1_handball_frames,
        'possible_frames': c1_possible_frames,
        'verdict': (
            "HANDBALL DETECTED" if c1_handball_frames
            else "POSSIBLE HANDBALL" if c1_possible_frames
            else "NO HANDBALL"
        ),
    }
    case2 = {
        'possible_foul_frames': c2_possible_foul_frames,
        'debatable_frames': c2_debatable_frames,
        'no_foul_frames': c2_no_foul_frames,
        'reaction_time_log': c2_reaction_times,
        'verdict': (
            "POSSIBLE FOUL" if c2_possible_foul_frames
            else "DEBATABLE" if c2_debatable_frames
            else "NO FOUL"
        ),
    }
    case3 = {
        'deliberate_frames': c3_deliberate_frames,
        'possible_frames': c3_possible_frames,
        'unlikely_frames': c3_unlikely_frames,
        'contact_events': c3_contact_events,
        'verdict': (
            "DELIBERATE HANDBALL" if c3_deliberate_frames
            else "POSSIBLE HANDBALL" if c3_possible_frames
            else "NO FOUL"
        ),
    }

    # ------------------------------------------------------------------
    # Combine into final verdict
    # ------------------------------------------------------------------
    final_verdict, is_foul, reason = _compute_final_verdict(case1, case2, case3, frame_count)

    result = {
        'final_verdict': final_verdict,
        'is_foul': is_foul,
        'reason': reason,
        'case1': case1,
        'case2': case2,
        'case3': case3,
        'total_frames': frame_count,
        'fps': fps,
    }

    if verbose:
        print("\n" + "=" * 60)
        _print_summary(result)
        print("=" * 60)

    return result


# =============================================================================
# FINAL VERDICT LOGIC
# =============================================================================

def _compute_final_verdict(case1, case2, case3, total_frames):
    """
    Aggregate the three case verdicts into one final decision.

    Hierarchy (most severe wins):
      1. Case 3 DELIBERATE HANDBALL  → definite foul
      2. Case 1 HANDBALL DETECTED    → definite foul (arm unnatural + contact)
      3. Case 3 POSSIBLE HANDBALL + Case 1 POSSIBLE/HANDBALL  → likely foul
      4. Case 2 POSSIBLE FOUL + Case 1 POSSIBLE/HANDBALL      → likely foul
      5. Multiple "possible" signals from different cases       → likely foul
      6. Single "possible" signal                              → debatable
      7. Everything clear                                      → no foul

    Returns (verdict_str, is_foul_bool, reason_str).
    """
    reasons = []

    # Severity scores per case (0 = clean, 1 = possible, 2 = detected/deliberate)
    s1 = 2 if case1['verdict'] == "HANDBALL DETECTED" else (1 if "POSSIBLE" in case1['verdict'] else 0)
    s2 = 2 if case2['verdict'] == "POSSIBLE FOUL" else (1 if case2['verdict'] == "DEBATABLE" else 0)
    s3 = 2 if case3['verdict'] == "DELIBERATE HANDBALL" else (1 if "POSSIBLE" in case3['verdict'] else 0)

    total_severity = s1 + s2 + s3

    # ---- Case 3 deliberate → strongest signal ----
    if s3 == 2:
        reasons.append(
            f"Case 3: Deliberate arm movement toward the ball detected in "
            f"{len(case3['deliberate_frames'])} frame(s), with ball deflection confirmed."
        )
    elif s3 == 1:
        reasons.append(
            f"Case 3: Arm was moving toward the ball in "
            f"{len(case3['possible_frames'])} frame(s), but no clear ball deflection."
        )
    else:
        reasons.append("Case 3: No deliberate arm movement toward the ball detected.")

    # ---- Case 1 ----
    if s1 == 2:
        reasons.append(
            f"Case 1: Arm made the body unnaturally bigger (high arm-torso angle + ball contact) "
            f"in {len(case1['handball_frames'])} frame(s)."
        )
    elif s1 == 1:
        reasons.append(
            f"Case 1: Arm was in a moderately extended position with ball nearby "
            f"in {len(case1['possible_frames'])} frame(s)."
        )
    else:
        reasons.append("Case 1: Arm position was natural (close to body) throughout.")

    # ---- Case 2 ----
    if s2 == 2:
        reasons.append(
            f"Case 2: Player had sufficient reaction time (> {REACTION_TIME_DEBATABLE} ms) "
            f"in {len(case2['possible_foul_frames'])} frame(s)."
        )
    elif s2 == 1:
        reasons.append(
            f"Case 2: Borderline reaction time ({REACTION_TIME_NO_FOUL}–{REACTION_TIME_DEBATABLE} ms) "
            f"in {len(case2['debatable_frames'])} frame(s)."
        )
    else:
        reasons.append(
            f"Case 2: Player had insufficient time to react (< {REACTION_TIME_NO_FOUL} ms) — no foul by reaction time."
        )

    reason_text = " ".join(reasons)

    # ---- Determine final verdict ----
    if s3 == 2 or (s1 == 2 and s2 >= 1):
        return "HANDBALL FOUL", True, reason_text

    if s1 == 2:
        # Case 1 strong but case 2 says no time to react
        if s2 == 0:
            return "HANDBALL UNLIKELY (no reaction time)", False, reason_text
        return "HANDBALL FOUL", True, reason_text

    if total_severity >= 4:
        return "HANDBALL FOUL", True, reason_text

    if total_severity >= 3:
        return "HANDBALL LIKELY", True, reason_text

    if total_severity >= 2:
        return "HANDBALL POSSIBLE — NEEDS REVIEW", False, reason_text

    if total_severity == 1:
        return "HANDBALL UNLIKELY", False, reason_text

    return "NO HANDBALL", False, reason_text


# =============================================================================
# ANNOTATION & PRINTING HELPERS
# =============================================================================

def _draw_combined_banner(frame, c1, c2, c3):
    """Draw a compact combined-verdict banner on the top of the annotated frame."""
    lines = []
    colors = []
    if c1 == "HANDBALL DETECTED":
        lines.append("C1: ARM UNNATURAL + CONTACT")
        colors.append((0, 0, 255))
    elif c1 == "POSSIBLE HANDBALL":
        lines.append("C1: POSSIBLE ARM ISSUE")
        colors.append((0, 165, 255))

    if c2 == "POSSIBLE FOUL":
        lines.append("C2: PLAYER HAD TIME")
        colors.append((0, 0, 255))
    elif c2 == "DEBATABLE":
        lines.append("C2: BORDERLINE REACTION")
        colors.append((0, 165, 255))

    if c3 == "DELIBERATE HANDBALL":
        lines.append("C3: DELIBERATE MOVEMENT + DEFLECTION")
        colors.append((0, 0, 255))
    elif c3 == "POSSIBLE HANDBALL":
        lines.append("C3: ARM TOWARD BALL")
        colors.append((0, 165, 255))
    elif c3 == "UNLIKELY FOUL":
        lines.append("C3: UNLIKELY FOUL")
        colors.append((0, 200, 255))

    y = 30
    for text, color in zip(lines, colors):
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 25


def _print_summary(result):
    """Print a human-readable summary of the analysis results."""
    tf = result['total_frames']
    c1 = result['case1']
    c2 = result['case2']
    c3 = result['case3']

    print(f"\nFINAL VERDICT:  {result['final_verdict']}")
    print(f"Is Foul:        {result['is_foul']}")
    print(f"Total frames:   {tf}")
    print()

    print("--- Case 1: Arm Unnaturally Bigger ---")
    print(f"  Verdict:         {c1['verdict']}")
    print(f"  Handball frames: {len(c1['handball_frames'])}")
    print(f"  Possible frames: {len(c1['possible_frames'])}")

    print("--- Case 2: Reaction Time ---")
    print(f"  Verdict:              {c2['verdict']}")
    print(f"  Possible foul frames: {len(c2['possible_foul_frames'])}")
    print(f"  Debatable frames:     {len(c2['debatable_frames'])}")
    print(f"  No foul frames:       {len(c2['no_foul_frames'])}")

    print("--- Case 3: Deliberate Arm Movement ---")
    print(f"  Verdict:            {c3['verdict']}")
    print(f"  Deliberate frames:  {len(c3['deliberate_frames'])}")
    print(f"  Possible frames:    {len(c3['possible_frames'])}")
    print(f"  Unlikely frames:    {len(c3['unlikely_frames'])}")

    print()
    print("REASON:")
    print(f"  {result['reason']}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python handball_foul.py <video_path> [output_path]")
        print("  video_path  — path to the input football video")
        print("  output_path — (optional) path for annotated output video")
        sys.exit(1)

    video = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isfile(video):
        print(f"Error: file not found — {video}")
        sys.exit(1)

    result = analyze_handball(video, output_path=out_path)
    print(f"\n{'='*40}")
    print(f"VERDICT: {result['final_verdict']}")
    print(f"IS FOUL: {result['is_foul']}")
    print(f"{'='*40}")
