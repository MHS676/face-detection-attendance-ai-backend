"""
Real-Time CCTV Object Detection using Roboflow Inference SDK
=============================================================

This script connects to a live CCTV camera via RTSP stream and performs
real-time object detection using Roboflow's Inference SDK with direct
OpenCV capture for reliable video display.

Features:
- Direct OpenCV capture for reliable RTSP streaming
- Bounding box visualization with supervision
- Real-time FPS counter
- Auto-reconnect on stream failure
- Graceful exit with 'q' key
"""

import cv2
import time
import os
import numpy as np
from collections import deque

# Suppress unnecessary model dependency warnings before importing inference
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["QWEN_3_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

# Roboflow Inference SDK for object detection
from inference import get_model

# Supervision for drawing annotations
import supervision as sv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Roboflow API credentials (set via environment variable)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "cctv-person-detection-flhrx/1")

# RTSP Stream Configuration — loaded from environment variables
# Set DVR_USER, DVR_PASS, DVR_HOST, DVR_PORT in your .env file
_dvr_user = os.getenv("DVR_USER", "admin")
_dvr_pass = os.getenv("DVR_PASS", "")
_dvr_host = os.getenv("DVR_HOST", "192.168.0.29")
_dvr_port = os.getenv("DVR_PORT", "554")
_dvr2_host = os.getenv("DVR2_HOST", "192.168.0.50")
_dvr2_port = os.getenv("DVR2_PORT", "8554")

RTSP_URLS = [
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/user={_dvr_user}&password={_dvr_pass}&channel=1&stream=0.sdp?",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/01",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/stream1",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr2_host}:{_dvr2_port}/user={_dvr_user}&password={_dvr_pass}&channel=1&stream=0.sdp?",
    f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr2_host}:{_dvr2_port}/01",
]

# Display settings
WINDOW_NAME = "CCTV Object Detection"
FPS_AVERAGING_WINDOW = 30

# =============================================================================
# ANNOTATION SETUP
# =============================================================================

# BoundingBoxAnnotator draws rectangles around detected objects
box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.ColorPalette.DEFAULT,
)

# LabelAnnotator displays class names and confidence scores
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=5,
    color=sv.ColorPalette.DEFAULT,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_fps(frame, fps: float) -> None:
    """Draw the FPS counter on the top-left of the frame."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.8, 2)
    x, y, p = 10, 30, 5
    cv2.rectangle(frame, (x - p, y - th - p), (x + tw + p, y + bl + p), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, 0.8, (0, 255, 0), 2)


def draw_detection_count(frame, count: int) -> None:
    """Draw the detection count below the FPS counter."""
    text = f"Detections: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.7, 2)
    x, y, p = 10, 65, 5
    cv2.rectangle(frame, (x - p, y - th - p), (x + tw + p, y + bl + p), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, 0.7, (255, 255, 0), 2)


def draw_status(frame, text: str, color=(0, 0, 255)) -> None:
    """Draw a centered status message on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    h, w = frame.shape[:2]
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), font, 1.0, color, 2)


def open_stream(url):
    """Open an RTSP stream with optimized settings. Returns cv2.VideoCapture."""
    # Try TCP transport first (more reliable than UDP for most cameras)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    if cap.isOpened():
        return cap

    # Fallback: try UDP transport
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;5000000"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        return cap

    # Final fallback: default backend
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main entry point for the CCTV object detection application."""

    print("\n" + "=" * 60)
    print("  CCTV Object Detection System")
    print("=" * 60)
    print(f"  Model : {MODEL_ID}")
    print(f"  URLs to try: {len(RTSP_URLS)}")
    print("  Press 'q' to quit")
    print("=" * 60)

    # --- Step 1: Create the display window FIRST so user sees something ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # Show a loading screen while model loads
    loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_status(loading_frame, "Loading model...", (255, 255, 255))
    cv2.imshow(WINDOW_NAME, loading_frame)
    cv2.waitKey(1)

    # --- Step 2: Load the Roboflow model ---
    print("\n[INFO] Loading model... (this may take a moment on first run)")
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    print("[INFO] Model loaded successfully!")

    # --- Step 3: Try each RTSP URL until one works ---
    cap = None
    working_url = None

    for i, url in enumerate(RTSP_URLS):
        status_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_status(status_frame, f"Trying URL {i+1}/{len(RTSP_URLS)}...", (255, 255, 255))
        cv2.imshow(WINDOW_NAME, status_frame)
        cv2.waitKey(1)

        print(f"[INFO] Trying ({i+1}/{len(RTSP_URLS)}): {url}")
        test_cap = open_stream(url)

        if test_cap.isOpened():
            # Try to actually read a frame to confirm it works
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                print(f"[OK] Connected and receiving frames!")
                cap = test_cap
                working_url = url
                break
            else:
                print(f"[WARN] Opened but no frames received, trying next...")
                test_cap.release()
        else:
            print(f"[FAIL] Could not open stream")

    if cap is None:
        print("\n[ERROR] None of the RTSP URLs worked!")
        print("[INFO] Trying local webcam as fallback...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            draw_status(error_frame, "No video source! Press 'q' to exit.", (0, 0, 255))
            cv2.imshow(WINDOW_NAME, error_frame)
            while True:
                if cv2.waitKey(100) & 0xFF in (ord('q'), ord('Q')):
                    break
            cv2.destroyAllWindows()
            return
        working_url = "webcam"

    print(f"\n[INFO] Using: {working_url}")
    print("[INFO] Detection running — window should now be visible.\n")

    # --- Step 4: Main detection loop ---
    frame_times = deque(maxlen=FPS_AVERAGING_WINDOW)
    frame_count = 0
    reconnect_attempts = 0
    max_reconnect = 5

    try:
        while True:
            ret, frame = cap.read()

            # Handle dropped frames / disconnection
            if not ret or frame is None:
                reconnect_attempts += 1
                print(f"[WARN] Frame lost. Reconnect attempt {reconnect_attempts}/{max_reconnect}...")

                # Show reconnecting screen
                recon_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                draw_status(recon_frame, f"Reconnecting... ({reconnect_attempts}/{max_reconnect})")
                cv2.imshow(WINDOW_NAME, recon_frame)

                if reconnect_attempts > max_reconnect:
                    print("[ERROR] Max reconnect attempts reached. Exiting.")
                    break

                cap.release()
                time.sleep(2)
                cap = open_stream(working_url)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                continue

            # Reset reconnect counter on success
            reconnect_attempts = 0

            # FPS tracking
            frame_times.append(time.time())
            frame_count += 1

            # --- Run object detection on the frame ---
            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            # --- Annotate the frame ---
            if len(detections) > 0:
                labels = [
                    f"{cn} {c:.2f}"
                    for cn, c in zip(detections['class_name'], detections.confidence)
                ]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # --- Draw HUD ---
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0

            draw_fps(frame, fps)
            draw_detection_count(frame, len(detections))

            # --- Display the annotated frame ---
            cv2.imshow(WINDOW_NAME, frame)

            # --- Check for 'q' to quit ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n[INFO] 'q' pressed — shutting down gracefully...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — shutting down...")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_count} frames total.")
        print("[INFO] Application terminated.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
