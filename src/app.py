import streamlit as st
import cv2
import mediapipe as mp
import imutils
import numpy as np
import time
from collections import deque
from stress_analyzer import StressAnalyzer
import uuid

# =============================
# Streamlit Page Configuration
# =============================
st.set_page_config(page_title="Eye Stress Detector AI", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0b0e11; color: #f0f0f0; }
    .stApp { background-color: #0b0e11; }
    h1, h2, h3, h4 { color: #ffffff !important; }
    textarea { font-family: monospace; color: #00ff80 !important; background-color: #0b0e11 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Layout
# =============================
st.markdown("## Eye Stress Detector AI")
st.divider()
col1, col2, col3 = st.columns([1.2, 2, 1.2])
log_box = col1.empty()
frame_window = col2.empty()
analysis_box = col3.empty()
st.divider()
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

# =============================
# MediaPipe Setup
# =============================
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(pts):
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

def extract_eye_points(landmarks, w, h, idxs):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]


# =============================
# Initialize Persistent State
# =============================
if "logs" not in st.session_state:
    st.session_state.logs = deque(maxlen=50)
if "total_blinks" not in st.session_state:
    st.session_state.total_blinks = 0
if "baseline" not in st.session_state:
    st.session_state.baseline = None
if "ema_ear" not in st.session_state:
    st.session_state.ema_ear = None
if "state_closed" not in st.session_state:
    st.session_state.state_closed = False
if "closed_start_time" not in st.session_state:
    st.session_state.closed_start_time = None
if "last_blink_time" not in st.session_state:
    st.session_state.last_blink_time = 0.0
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "stress" not in st.session_state:
    st.session_state.stress = StressAnalyzer(window_sec=10)
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()
if "ear_values" not in st.session_state:
    st.session_state.ear_values = []
if "final_metrics" not in st.session_state:
    st.session_state.final_metrics = None

# Tunables
LOW_FRAC, HIGH_FRAC, EMA_ALPHA = 0.78, 0.84, 0.4
MIN_CLOSED_MS, REFRACTORY_MS = 60, 200
BASELINE_UPDATE_INTERVAL = 30
UPDATE_INTERVAL = 10  # seconds

# =============================
# Camera Toggle
# =============================
run = st.toggle("Turn Camera On", False)

if run:
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        st.info("Camera started. Keep eyes open for calibration...")

        while run:
            ok, frame = cap.read()
            if not ok:
                st.error("Camera feed not available.")
                break

            frame = imutils.resize(frame, width=720)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            now = time.time()
            just_blinked = False

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                left_pts = extract_eye_points(face, w, h, LEFT_EYE_IDX)
                right_pts = extract_eye_points(face, w, h, RIGHT_EYE_IDX)

                leftEAR = eye_aspect_ratio(left_pts)
                rightEAR = eye_aspect_ratio(right_pts)
                ear = (leftEAR + rightEAR) / 2.0

                # Smooth EAR
                if st.session_state.ema_ear is None:
                    st.session_state.ema_ear = ear
                else:
                    st.session_state.ema_ear = EMA_ALPHA * ear + (1 - EMA_ALPHA) * st.session_state.ema_ear

                st.session_state.ear_values.append(st.session_state.ema_ear)

                # Draw eyes
                cv2.polylines(frame, [np.array(left_pts)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_pts)], True, (0, 255, 0), 1)

                # --- Baseline Calibration ---
                if st.session_state.baseline is None:
                    st.session_state.baseline = st.session_state.ema_ear
                    st.session_state.stress._baseline_ear = st.session_state.baseline
                    st.session_state.logs.append("[INFO] Calibrating... keep eyes open.")
                else:
                    low_thr = st.session_state.baseline * LOW_FRAC
                    high_thr = st.session_state.baseline * HIGH_FRAC

                    # --- Blink Detection ---
                    if not st.session_state.state_closed:
                        if st.session_state.ema_ear < low_thr:
                            st.session_state.state_closed = True
                            st.session_state.closed_start_time = now
                    else:
                        if st.session_state.ema_ear > high_thr:
                            closed_ms = (now - st.session_state.closed_start_time) * 1000.0 if st.session_state.closed_start_time else 0.0
                            gap_ms = (now - st.session_state.last_blink_time) * 1000.0
                            if closed_ms >= MIN_CLOSED_MS and gap_ms >= REFRACTORY_MS:
                                st.session_state.total_blinks += 1
                                st.session_state.last_blink_time = now
                                just_blinked = True
                                st.session_state.logs.append(f"[EVENT] Blink #{st.session_state.total_blinks}")
                            st.session_state.state_closed = False
                            st.session_state.closed_start_time = None

                    # --- Update Analyzer ---
                    st.session_state.stress.update(st.session_state.ema_ear, just_blinked)

                    # --- Compute Stress Every 10s ---
                    if now - st.session_state.last_update_time >= UPDATE_INTERVAL:
                        result = st.session_state.stress.compute()
                        if result:
                            st.session_state.last_result = result
                            st.session_state.logs.append(f"[STRESS] {result}")
                        st.session_state.last_update_time = now

                # --- HUD Overlay ---
                cv2.putText(frame, f"Blinks: {st.session_state.total_blinks}", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"EAR: {st.session_state.ema_ear:.3f}", (200, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                if st.session_state.last_result:
                    stress_text = st.session_state.last_result["stress_indicator"]
                    color = (0, 255, 0)
                    if "Moderate" in stress_text:
                        color = (0, 255, 255)
                    elif "High" in stress_text:
                        color = (0, 0, 255)
                    cv2.putText(frame, f"Stress: {stress_text}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # --- Stable Averages (every 10s) ---
                if st.session_state.last_result:
                    avg_ear = np.mean(st.session_state.ear_values) if st.session_state.ear_values else 0
                    stress_level = st.session_state.last_result["stress_indicator"]
                    stress_score = st.session_state.last_result["stress_score"]
                    confidence = round((1 - abs(avg_ear - st.session_state.baseline) / st.session_state.baseline) * 100, 1)

                    analysis_box.markdown(f"""
                    ### AI Analysis & Conclusion
                    - **Total Blinks**: {st.session_state.total_blinks}
                    - **Average EAR**: {avg_ear:.3f}
                    - **Stress Level**: {stress_level}
                    - **Score**: {stress_score:.2f}
                    """)

                    metrics_col1.metric("Total Blinks", st.session_state.total_blinks)
                    metrics_col2.metric("Avg EAR", f"{avg_ear:.3f}")
                    metrics_col3.metric("Stress Level", stress_level)
                    metrics_col4.metric("Confidence", f"{confidence:.1f}%")

            else:
                st.session_state.logs.append("[WARN] No face detected.")
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # --- Log Terminal ---
            formatted_logs = "\n".join(list(st.session_state.logs))
            log_box.text_area("EYE-TRACK_LOGS", formatted_logs, height=450, key=str(uuid.uuid4()))

            time.sleep(0.05)

    # Store final metrics when session ends
    if st.session_state.last_result:
        avg_ear = np.mean(st.session_state.ear_values) if st.session_state.ear_values else 0
        stress_level = st.session_state.last_result["stress_indicator"]
        stress_score = st.session_state.last_result["stress_score"]
        confidence = round((1 - abs(avg_ear - st.session_state.baseline) / st.session_state.baseline) * 100, 1)
        
        st.session_state.final_metrics = {
            "total_blinks": st.session_state.total_blinks,
            "avg_ear": avg_ear,
            "stress_level": stress_level,
            "stress_score": stress_score,
            "confidence": confidence
        }

    cap.release()

else:
    # Display final metrics in cards if session has ended and metrics are available
    if st.session_state.final_metrics:
        metrics = st.session_state.final_metrics
        metrics_col1.metric("Total Blinks", metrics["total_blinks"])
        metrics_col2.metric("Avg EAR", f"{metrics['avg_ear']:.3f}")
        metrics_col3.metric("Stress Level", metrics["stress_level"])
        metrics_col4.metric("Confidence", f"{metrics['confidence']:.1f}%")
        
        # Clear analysis box
        analysis_box.markdown(f"""
        ### Session Results
        - **Total Blinks**: {metrics["total_blinks"]}
        - **Average EAR**: {metrics["avg_ear"]:.3f}
        - **Stress Level**: {metrics["stress_level"]}
        - **Score**: {metrics["stress_score"]:.2f}
        """)
    else:
        st.info("Turn on the camera to begin live analysis.")
