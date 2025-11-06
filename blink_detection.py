import cv2
import time
import numpy as np
from collections import deque
import imutils
import mediapipe as mp

# ---- MediaPipe FaceMesh setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Eye landmark indices for EAR (MediaPipe FaceMesh 468-landmark topology)
# We mirror the classic 6-point EAR definition: [p0,p1,p2,p3,p4,p5]
# Horizontal: p0—p3, Verticals: (p1—p5) and (p2—p4)
LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]   # [outer, up1, low1, inner, low2, up2]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373] # [outer, up1, low1, inner, low2, up2]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(pts):  # pts: list of 6 (x,y) pixel coords
    A = euclidean(pts[1], pts[5])  # vertical pair 1 (up1 <-> up2)
    B = euclidean(pts[2], pts[4])  # vertical pair 2 (low1 <-> low2)
    C = euclidean(pts[0], pts[3])  # horizontal (outer <-> inner)
    if C <= 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)

def extract_eye_points(landmarks, image_w, image_h, idxs):
    pts = []
    for i in idxs:
        lm = landmarks[i]
        # convert normalized -> pixel to keep x/y scales correct
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        pts.append((x, y))
    return pts

def main():
    # ---- Tunables ----
    DISPLAY_WIDTH = 720
    CALIBRATION_SECONDS = 3.0
    EMA_ALPHA = 0.35

    LOW_FRAC = 0.72         # closed threshold = baseline * LOW_FRAC
    HIGH_FRAC = 0.82        # reopen threshold = baseline * HIGH_FRAC
    MIN_CLOSED_MS = 90      # must be closed at least this long
    REFRACTORY_MS = 220     # min gap between blink counts
    BLINK_RATE_WINDOW = 60  # seconds (display info overlay)

    # ---- State ----
    total_blinks = 0
    last_blink_time = 0.0
    closed_start_time = None
    state_closed = False
    baseline = None
    ema_ear = None
    calib_buf = deque()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    time.sleep(0.5)

    start_time = time.time()
    last_rate_time = start_time

    # Use FaceMesh with attention to stability
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,   # refines eyes/iris
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = imutils.resize(frame, width=DISPLAY_WIDTH)
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            now = time.time()

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark

                left_pts = extract_eye_points(face, w, h, LEFT_EYE_IDX)
                right_pts = extract_eye_points(face, w, h, RIGHT_EYE_IDX)

                leftEAR = eye_aspect_ratio(left_pts)
                rightEAR = eye_aspect_ratio(right_pts)
                ear = (leftEAR + rightEAR) / 2.0

                # Smooth EAR
                if ema_ear is None:
                    ema_ear = ear
                else:
                    ema_ear = EMA_ALPHA * ear + (1 - EMA_ALPHA) * ema_ear

                # Draw eye contours
                cv2.polylines(frame, [np.array(left_pts, dtype=np.int32)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_pts, dtype=np.int32)], True, (0, 255, 0), 1)

                # --- Calibration (keep eyes open) ---
                elapsed = now - start_time
                if baseline is None:
                    calib_buf.append(ema_ear)
                    cv2.putText(frame, "Calibrating... keep eyes open",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"{elapsed:.1f}s / {CALIBRATION_SECONDS}s",
                                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    if elapsed >= CALIBRATION_SECONDS and len(calib_buf) > 10:
                        arr = np.array(calib_buf)
                        # robust “open-eye” baseline
                        baseline = float(np.percentile(arr, 95))
                        # quick sanity clamps
                        baseline = np.clip(baseline, 0.15, 0.45)

                    cv2.imshow("MediaPipe Blink Detection (Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                low_thr = baseline * LOW_FRAC
                high_thr = max(baseline * HIGH_FRAC, low_thr + 0.02)

                # --- Hysteresis state machine ---
                if not state_closed:
                    if ema_ear < low_thr:
                        state_closed = True
                        closed_start_time = now
                else:
                    if ema_ear > high_thr:
                        closed_ms = (now - closed_start_time) * 1000.0 if closed_start_time else 0.0
                        gap_ms = (now - last_blink_time) * 1000.0
                        if closed_ms >= MIN_CLOSED_MS and gap_ms >= REFRACTORY_MS:
                            total_blinks += 1
                            last_blink_time = now
                        state_closed = False
                        closed_start_time = None

                # --- Blink rate overlay (overall so far) ---
                if now - last_rate_time >= BLINK_RATE_WINDOW:
                    minutes = (now - start_time) / 60.0
                    blink_rate = total_blinks / minutes if minutes > 0 else 0.0
                    stress = "Low Stress" if blink_rate <= 25 else "High Stress/Fatigue"
                    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min",
                                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"Stress Level: {stress}",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    last_rate_time = now

                # --- HUD ---
                cv2.putText(frame, f"Blinks: {total_blinks}", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"EAR: {ema_ear:.3f}", (200, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(frame, f"thr_low:{low_thr:.3f} thr_high:{high_thr:.3f}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
                if state_closed:
                    cv2.putText(frame, "Closed", (w - 140, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            else:
                # No face tracked this frame — keep state but show notice
                cv2.putText(frame, "No face detected", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("MediaPipe Blink Detection (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
