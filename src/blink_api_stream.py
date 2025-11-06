from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import cv2, time, imutils, numpy as np, threading
import mediapipe as mp
from stress_analyzer import StressAnalyzer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global shared state ---
is_running = False
frame_lock = threading.Lock()
output_frame = None
status_data = {
    "total_blinks": 0,
    "stress_level": "N/A",
    "stress_score": 0.0,
    "confidence": 0.0,
}

# --- Core analysis thread ---
def video_analysis():
    global output_frame, is_running, status_data

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    stress = StressAnalyzer(window_sec=10)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        is_running = False
        return
    time.sleep(0.3)

    baseline, ema_ear = None, None
    LOW_FRAC, HIGH_FRAC, EMA_ALPHA = 0.72, 0.82, 0.35
    total_blinks, closed, last_blink_time = 0, False, 0.0
    MIN_CLOSED_MS, REFRACTORY_MS = 100, 200
    start_close = 0.0

    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = imutils.resize(frame, width=720)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        h, w = frame.shape[:2]
        now = time.time()
        just_blinked = False

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0].landmark
            left_idx = [33, 159, 145, 133, 153, 144]
            right_idx = [362, 386, 374, 263, 380, 373]

            def ear(idxs):
                pts = [(int(face[i].x * w), int(face[i].y * h)) for i in idxs]
                A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

            leftEAR, rightEAR = ear(left_idx), ear(right_idx)
            avg_ear = (leftEAR + rightEAR) / 2.0
            ema_ear = avg_ear if ema_ear is None else EMA_ALPHA * avg_ear + (1 - EMA_ALPHA) * ema_ear

            # Initialize baseline
            if baseline is None:
                baseline = ema_ear
                stress._baseline_ear = baseline

            low_thr, high_thr = baseline * LOW_FRAC, baseline * HIGH_FRAC

            # Blink logic (with refractory)
            if not closed and ema_ear < low_thr:
                closed, start_close = True, now
            elif closed and ema_ear > high_thr:
                closed_ms = (now - start_close) * 1000
                gap_ms = (now - last_blink_time) * 1000
                if closed_ms >= MIN_CLOSED_MS and gap_ms >= REFRACTORY_MS:
                    total_blinks += 1
                    last_blink_time = now
                    just_blinked = True
                closed = False

            # Update stress analyzer
            stress.update(ema_ear, just_blinked)
            result = stress.compute()
            if result:
                confidence = round(
                    max(0.0, min(100.0, (1 - abs(result["avg_ear"] - baseline) / max(baseline, 1e-6)) * 100)),
                    1,
                )

                status_data.update({
                    "total_blinks": total_blinks,
                    "stress_level": result["stress_indicator"],
                    "stress_score": round(result["stress_score"], 2),
                    "confidence": confidence,
                })

            # HUD overlays
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ema_ear:.3f}", (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Stress: {status_data['stress_level']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {status_data['confidence']}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode to JPEG
        with frame_lock:
            _, jpeg = cv2.imencode(".jpg", frame)
            output_frame = jpeg.tobytes()

        time.sleep(0.03)  # ~30 FPS

    cap.release()
    face_mesh.close()
    print("[INFO] Camera stream stopped")

# --- MJPEG generator for React <img> ---
def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        time.sleep(0.03)

# --- API endpoints ---
@app.post("/start")
def start(background_tasks: BackgroundTasks):
    global is_running
    if not is_running:
        is_running = True
        background_tasks.add_task(video_analysis)
        return {"status": "started"}
    return {"status": "already running"}

@app.post("/stop")
def stop():
    global is_running
    is_running = False
    return {"status": "stopped", "result": status_data}

@app.get("/status")
def get_status():
    return status_data

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
