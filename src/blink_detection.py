import cv2
import time
import numpy as np
from collections import deque
import imutils
import mediapipe as mp
from stress_analyzer import StressAnalyzer   # <<— imported here

# ---- MediaPipe setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

def main():
    DISPLAY_WIDTH = 720
    CALIBRATION_SECONDS = 3.0
    EMA_ALPHA = 0.35
    LOW_FRAC, HIGH_FRAC = 0.72, 0.82
    MIN_CLOSED_MS, REFRACTORY_MS = 90, 220

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

    stress = StressAnalyzer(window_sec=10)  # initialize analyzer

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
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

                ema_ear = ear if ema_ear is None else EMA_ALPHA * ear + (1 - EMA_ALPHA) * ema_ear

                cv2.polylines(frame, [np.array(left_pts)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_pts)], True, (0, 255, 0), 1)

                # Calibration
                elapsed = now - start_time
                if baseline is None:
                    calib_buf.append(ema_ear)
                    cv2.putText(frame, "Calibrating... keep eyes open",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if elapsed >= CALIBRATION_SECONDS and len(calib_buf) > 10:
                        arr = np.array(calib_buf)
                        baseline = np.clip(np.percentile(arr, 95), 0.15, 0.45)
                    cv2.imshow("Blink Detection (Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                low_thr, high_thr = baseline * LOW_FRAC, baseline * HIGH_FRAC
                stress._baseline_ear = baseline  # pass your personal open-eye baseline

                # Blink detection
                if not state_closed and ema_ear < low_thr:
                    state_closed, closed_start_time = True, now
                elif state_closed and ema_ear > high_thr:
                    closed_ms = (now - closed_start_time) * 1000 if closed_start_time else 0
                    gap_ms = (now - last_blink_time) * 1000
                    if closed_ms >= MIN_CLOSED_MS and gap_ms >= REFRACTORY_MS:
                        total_blinks += 1
                        last_blink_time = now
                        just_blinked = True
                    state_closed, closed_start_time = False, None

                # Update stress analyzer
                stress.update(ema_ear, just_blinked)
                result = stress.compute()

                if result:
                    color = (0, 255, 0)
                    if result["stress_indicator"] == "Moderate Stress":
                        color = (0, 255, 255)
                    elif "High" in result["stress_indicator"]:
                        color = (0, 0, 255)

                    cv2.putText(frame, f"Blink Rate: {result['blink_rate']:.1f}/min", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Stress: {result['stress_indicator']}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    print("[STRESS]", result)

                cv2.putText(frame, f"Blinks: {total_blinks}", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"EAR: {ema_ear:.3f}", (200, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                if state_closed:
                    cv2.putText(frame, "Closed", (w - 140, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("Blink Detection (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
