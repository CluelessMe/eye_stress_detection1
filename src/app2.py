import cv2
import time
import numpy as np
import threading
from flask import Flask, Response, jsonify, send_from_directory
import mediapipe as mp

# IMPORTANT: Enable static folder serving
app = Flask(__name__, static_folder="static")

# =========================
# MediaPipe Face Mesh Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(pts):
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

# =========================
# Shared State
# =========================
state = {
    "ear": 0.0,
    "blinks": 0,
    "stress": "Calibrating...",
    "logs": []
}

current_frame = None
running = True


# =========================
# Camera Processing Loop
# =========================
def camera_loop():
    global current_frame

    cap = cv2.VideoCapture(0)

    baseline, ema = None, None
    alpha = 0.4
    LOW_FRAC, HIGH_FRAC = 0.78, 0.84
    MIN_CLOSED_MS, REFRACTORY_MS = 60, 200
    closed = False
    closed_start = 0
    last_blink_time = 0

    while running:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        now = time.time()

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def get_pts(idx_list):
                return [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_list]

            left, right = get_pts(LEFT_EYE_IDX), get_pts(RIGHT_EYE_IDX)
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ema = ear if ema is None else alpha * ear + (1 - alpha) * ema
            state["ear"] = round(ema, 3)

            if baseline is None:
                baseline = ema
                state["logs"].append("[INFO] Calibrating baseline")
            else:
                low_thr = baseline * LOW_FRAC
                high_thr = baseline * HIGH_FRAC

                if not closed and ema < low_thr:
                    closed = True
                    closed_start = now

                elif closed and ema > high_thr:
                    closed_ms = (now - closed_start) * 1000
                    gap = (now - last_blink_time) * 1000

                    if closed_ms >= MIN_CLOSED_MS and gap >= REFRACTORY_MS:
                        state["blinks"] += 1
                        last_blink_time = now
                        state["logs"].append(f"[BLINK] #{state['blinks']}")

                    closed = False

                if ema < baseline * 0.8:
                    state["stress"] = "High"
                elif ema < baseline * 0.9:
                    state["stress"] = "Moderate"
                else:
                    state["stress"] = "Normal"

            cv2.putText(frame, f"Blinks: {state['blinks']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"EAR: {state['ear']:.3f}", (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Stress: {state['stress']}", (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        current_frame = buffer.tobytes()

    cap.release()


threading.Thread(target=camera_loop, daemon=True).start()


# =========================
# UI Route (this fixes 404)
# =========================
@app.route("/")
def index():
    return app.send_static_file("index1.html")


# =========================
# MJPEG Stream Route
# =========================
def gen_frames():
    global current_frame
    while True:
        if current_frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current_frame + b"\r\n")
        else:
            time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame", direct_passthrough=True)


# =========================
# Metrics JSON API
# =========================
@app.route("/metrics")
def metrics():
    return jsonify({
        "ear": state["ear"],
        "blinks": state["blinks"],
        "stress": state["stress"],
        "logs": state["logs"][-12:]
    })


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

