"""
CCTV Attendance System — Flask Backend
=======================================
Features:
  • Roboflow Inference SDK for person detection
  • ByteTrack object tracking (supervision)
  • face_recognition for identity matching
  • In/Out counting with position-history direction analysis
  • Attendance logging with face identity
  • MJPEG live video feed with annotations
  • Known-face management (register / list / delete)
"""

import os, sys

# Suppress noisy inference SDK warnings before any imports
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["QWEN_3_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

import io, time, threading, base64, pickle
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Suppress numpy verbose printing (dlib dumps arrays to stdout)
np.set_printoptions(threshold=0, edgeitems=0)

# Ensure print output is flushed immediately (important for nohup)
import functools
print = functools.partial(print, flush=True)

# face_recognition (dlib-based)
import face_recognition

# Roboflow Inference
from inference import get_model

# supervision
import supervision as sv

# Local models
from models import (
    init_db, get_session,
    KnownFace, AttendanceLog, DetectionEvent, ObjectCount, CameraConfig,
)

# =============================================================================
# ENV
# =============================================================================
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "y6YIohxGFgARrNJJAJgK")
MODEL_ID = os.getenv("MODEL_ID", "cctv-person-detection-flhrx/1")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# ROBOFLOW MODEL (lazy singleton)
# =============================================================================
_model = None
_model_lock = threading.Lock()


def get_rf_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[MODEL] Loading Roboflow model {MODEL_ID} …")
                _model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
                print("[MODEL] Model loaded ✓")
    return _model


# =============================================================================
# KNOWN-FACE CACHE  (loaded once, refreshed on changes)
# =============================================================================
_known_encodings: list[np.ndarray] = []
_known_names: list[str] = []
_known_ids: list[int] = []
_faces_lock = threading.Lock()


def reload_known_faces():
    """Pull all active face encodings from DB into memory."""
    global _known_encodings, _known_names, _known_ids
    session = get_session()
    try:
        rows = session.query(KnownFace).filter_by(is_active=True).all()
        encs, names, ids = [], [], []
        for r in rows:
            enc = pickle.loads(r.encoding)  # np.ndarray 128-d
            encs.append(enc)
            names.append(r.name)
            ids.append(r.id)
        with _faces_lock:
            _known_encodings = encs
            _known_names = names
            _known_ids = ids
        print(f"[FACES] Loaded {len(encs)} known face(s)")
    finally:
        session.close()


# =============================================================================
# CAMERA STREAM MANAGER
# =============================================================================

class CameraStream:
    """
    Manages a single camera: capture → detect → track → identify → count.
    """

    FACE_MATCH_TOLERANCE = 0.68  # tolerance for CCTV + CNN model (farther, lower-res, angled faces)

    def __init__(self, channel: str, rtsp_url: str, name: str = ""):
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.name = name or channel

        # State
        self.cap: cv2.VideoCapture | None = None
        self.connected = False
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

        # Latest annotated frame (JPEG)
        self.frame_jpeg: bytes = b""
        self.frame_w = 0
        self.frame_h = 0

        # supervision tracker + annotators
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=4,
        )

        # In / Out counting — position-history approach
        self.in_count: dict[str, int] = defaultdict(int)   # class → count
        self.out_count: dict[str, int] = defaultdict(int)
        self._track_history: dict[int, deque] = {}          # tracker_id → deque of y-centers
        self._counted_ids: set[int] = set()                 # already counted

        # Per-tracker face identity
        self._tracker_identity: dict[int, str] = {}         # tracker_id → person_name
        self._tracker_face_conf: dict[int, float] = {}      # tracker_id → best distance
        self._tracker_face_attempts: dict[int, int] = {}    # tracker_id → how many face attempts
        self._logged_tracker_ids: dict[int, int] = {}       # tracker_id → AttendanceLog.id (for updating Unknown→name)

        # Detection history (in-memory buffer)
        self.detection_history: list[dict] = []
        self._history_lock = threading.Lock()

        # FPS limiter
        self._last_infer_time = 0
        self._infer_interval = 0.25  # seconds between inferences (4 FPS)
        self._face_interval = 3.0    # face recognition every 3.0 s (CNN model is heavy on CPU)
        self._last_face_time = 0

    # -----------------------------------------------------------------
    # Connect / Disconnect
    # -----------------------------------------------------------------
    def connect(self) -> bool:
        if self.connected:
            return True
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[CAM:{self.channel}] Cannot open {self.rtsp_url}")
                return False
            self.cap = cap
            self.connected = True
            self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
            self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
            print(f"[CAM:{self.channel}] Connected  {self.frame_w}×{self.frame_h}")
            return True
        except Exception as e:
            print(f"[CAM:{self.channel}] Connection error: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        print(f"[CAM:{self.channel}] Disconnected")

    # -----------------------------------------------------------------
    # Start / Stop processing
    # -----------------------------------------------------------------
    def start(self):
        if self.running:
            return
        if not self.connected and not self.connect():
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[CAM:{self.channel}] Processing started")

    def stop(self):
        self.running = False

    # -----------------------------------------------------------------
    # Main processing loop
    # -----------------------------------------------------------------
    def _loop(self):
        model = get_rf_model()

        while self.running and self.connected:
            if self.cap is None or not self.cap.isOpened():
                self._try_reconnect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._try_reconnect()
                continue

            now = time.time()
            run_infer = (now - self._last_infer_time) >= self._infer_interval
            run_face = (now - self._last_face_time) >= self._face_interval

            if run_infer:
                self._last_infer_time = now
                annotated = self._process_frame(frame, model, run_face)
                if run_face:
                    self._last_face_time = now
            else:
                annotated = frame

            # Encode to JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.frame_jpeg = buf.tobytes()

            # Tiny sleep to avoid busy-loop
            time.sleep(0.01)

    def _try_reconnect(self):
        print(f"[CAM:{self.channel}] Reconnecting …")
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.connected = False
            self.running = False
            print(f"[CAM:{self.channel}] Reconnection failed")

    # -----------------------------------------------------------------
    # Frame processing pipeline
    # -----------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, model, run_face: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        line_y = int(h * 0.55)  # counting line at 55 %

        # --- 1. Roboflow person detection ---
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        # --- 2. ByteTrack ---
        detections = self.tracker.update_with_detections(detections)

        # --- 3. Face recognition (periodically) ---
        if run_face and len(detections) > 0:
            self._run_face_recognition(frame, detections)

        # --- 4. In/Out counting via position history ---
        self._update_tracking_and_count(detections, line_y)

        # --- 5. Build labels ---
        labels = []
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                cls_name = (
                    detections.data.get("class_name", ["person"])[i]
                    if "class_name" in detections.data
                    else "person"
                )
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0
                identity = self._tracker_identity.get(int(tid), "")
                label = f"#{tid}"
                if identity:
                    label += f" {identity}"
                label += f" {conf:.0%}"
                labels.append(label)
        else:
            labels = [f"{detections.confidence[i]:.0%}" for i in range(len(detections))]

        # --- 6. Annotate ---
        annotated = frame.copy()

        # Draw counting line
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(annotated, "IN ^", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, "v OUT", (10, line_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        # HUD — totals
        total_in = sum(self.in_count.values())
        total_out = sum(self.out_count.values())
        cv2.putText(annotated, f"IN: {total_in}", (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"OUT: {total_out}", (w - 160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated

    # -----------------------------------------------------------------
    # Face recognition on detected persons
    # -----------------------------------------------------------------
    def _run_face_recognition(self, frame: np.ndarray, detections: sv.Detections):
        """For each detected person box, crop face area and try to match."""
        if detections.tracker_id is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with _faces_lock:
            known_encs = list(_known_encodings)
            known_names = list(_known_names)
            known_ids = list(_known_ids)

        if len(known_encs) == 0:
            return  # no known faces to match against

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)

            # Track how many times we've tried face recognition for this tracker
            attempts = self._tracker_face_attempts.get(tid, 0)

            # Skip if we have a CONFIRMED (non-Unknown) identity with good confidence
            current_identity = self._tracker_identity.get(tid, "")
            current_conf = self._tracker_face_conf.get(tid, 1.0)
            if current_identity and current_identity != "Unknown" and current_conf < 0.45:
                continue  # confident match already found

            # Keep retrying Unknown faces (up to 20 attempts)
            if current_identity == "Unknown" and attempts >= 20:
                continue  # give up after 20 attempts

            self._tracker_face_attempts[tid] = attempts + 1

            # Crop person bounding box
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            person_h = y2 - y1
            person_w = x2 - x1

            if attempts < 1:  # Log only first attempt per tracker
                print(f"[FACE-DBG] #{tid} attempt {attempts+1}: person bbox={person_w}x{person_h}")

            # Strategy 1: Upper 65% of person bbox (head + shoulders)
            face_y2 = y1 + int(person_h * 0.65)
            pad = 30
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(frame.shape[1], x2 + pad)
            cy2 = min(frame.shape[0], face_y2)

            best_name = None
            best_dist = 1.0

            # Try 2 crop strategies: upper 60% with padding, full person
            crop_regions = [
                # Crop A: Upper 60% with generous padding (head + shoulders)
                (cy1, cy2, cx1, cx2),
                # Crop B: Full person bbox (for overhead cameras where face is in middle)
                (max(0, y1 - 10), min(frame.shape[0], y2 + 10), max(0, x1 - 10), min(frame.shape[1], x2 + 10)),
            ]

            crop_idx = 0
            for crop_region in crop_regions:
                crop_idx += 1
                r_y1, r_y2, r_x1, r_x2 = crop_region
                if r_x2 - r_x1 < 20 or r_y2 - r_y1 < 20:
                    continue

                face_crop_rgb = rgb_frame[r_y1:r_y2, r_x1:r_x2]

                # Upscale small crops for better face detection
                crop_h, crop_w = face_crop_rgb.shape[:2]
                if crop_h < 150 or crop_w < 120:
                    scale = max(150 / max(crop_h, 1), 120 / max(crop_w, 1), 1.0)
                    if scale > 1.0:
                        face_crop_rgb = cv2.resize(
                            face_crop_rgb, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC
                        )

                # CLAHE enhancement (always apply for CCTV — essential for detection)
                enhanced_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
                lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                lab = cv2.merge((l_ch, a_ch, b_ch))
                enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

                # Try face detection — use CNN model directly (HOG fails on CCTV angles)
                face_locations = face_recognition.face_locations(enhanced_rgb, model="cnn")
                if len(face_locations) > 0:
                    face_crop_rgb = enhanced_rgb
                else:
                    # Fallback: HOG with upsample=2 on enhanced image
                    face_locations = face_recognition.face_locations(
                        enhanced_rgb, number_of_times_to_upsample=2, model="hog"
                    )
                    if len(face_locations) > 0:
                        face_crop_rgb = enhanced_rgb
                    else:
                        # Last resort: HOG on original
                        face_locations = face_recognition.face_locations(face_crop_rgb, model="hog")

                if len(face_locations) == 0:
                    if attempts < 1:
                        print(f"[FACE-DBG] #{tid} crop{crop_idx}: {face_crop_rgb.shape[1]}x{face_crop_rgb.shape[0]} — no face found (all strategies)")
                    continue

                try:
                    # Suppress dlib's noisy error output during encoding
                    old_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    try:
                        encodings = face_recognition.face_encodings(face_crop_rgb, face_locations)
                    finally:
                        sys.stderr.close()
                        sys.stderr = old_stderr
                except (TypeError, Exception):
                    continue
                if len(encodings) == 0:
                    continue

                if attempts < 1:
                    print(f"[FACE-DBG] #{tid} crop{crop_idx}: {len(face_locations)} face(s) found, encoding...")

                # Match against known faces (try all detected faces in crop)
                for enc in encodings:
                    distances = face_recognition.face_distance(known_encs, enc)
                    if len(distances) == 0:
                        continue
                    idx = int(np.argmin(distances))
                    dist = float(distances[idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_name = known_names[idx] if dist < self.FACE_MATCH_TOLERANCE else None

                if best_name and best_dist < self.FACE_MATCH_TOLERANCE:
                    break  # found a good match, no need to try second crop

            # Apply result
            if best_name and best_dist < self.FACE_MATCH_TOLERANCE:
                prev = self._tracker_face_conf.get(tid, 1.0)
                if best_dist < prev:
                    old = self._tracker_identity.get(tid, "")
                    self._tracker_identity[tid] = best_name
                    self._tracker_face_conf[tid] = best_dist
                    if old != best_name:
                        print(f"[FACE] #{tid} identified as '{best_name}' (dist={best_dist:.3f})")
                    # If this tracker was already counted as Unknown, update the DB record
                    if old == "Unknown" and tid in self._logged_tracker_ids:
                        threading.Thread(
                            target=self._update_attendance_identity,
                            args=(tid, best_name, best_dist),
                            daemon=True,
                        ).start()
                    # Also update in-memory detection history
                    if old == "Unknown":
                        with self._history_lock:
                            for evt in self.detection_history:
                                if evt.get("tracker_id") == tid and evt.get("person_name") == "Unknown":
                                    evt["person_name"] = best_name
                                    evt["confidence"] = round(best_dist, 3)
            else:
                if tid not in self._tracker_identity:
                    self._tracker_identity[tid] = "Unknown"
                    self._tracker_face_conf[tid] = best_dist

    # -----------------------------------------------------------------
    # Position-history-based In/Out counting
    # -----------------------------------------------------------------
    def _update_tracking_and_count(self, detections: sv.Detections, line_y: int):
        """
        Track the center-y of each detection over time.
        When a tracker's position crosses `line_y`:
          - bottom→top  ⇒  IN
          - top→bottom  ⇒  OUT
        """
        if detections.tracker_id is None:
            return

        active_ids = set()

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            active_ids.add(tid)

            x1, y1, x2, y2 = detections.xyxy[i]
            cy = (y1 + y2) / 2.0

            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=30)
            self._track_history[tid].append(cy)

            # Need at least a few frames of history
            hist = self._track_history[tid]
            if len(hist) < 5 or tid in self._counted_ids:
                continue

            # Check if the tracker crossed the line
            old_y = hist[0]   # oldest position
            new_y = hist[-1]  # newest position

            crossed_down = old_y < line_y and new_y >= line_y
            crossed_up = old_y > line_y and new_y <= line_y

            if not crossed_down and not crossed_up:
                continue

            cls_name = (
                detections.data.get("class_name", ["person"])[i]
                if "class_name" in detections.data
                else "person"
            )
            identity = self._tracker_identity.get(tid, "Unknown")
            conf = self._tracker_face_conf.get(tid, 0.0)

            if crossed_up:
                direction = "in"
                self.in_count[cls_name] += 1
            else:
                direction = "out"
                self.out_count[cls_name] += 1

            self._counted_ids.add(tid)

            print(f"[COUNT] #{tid} crossed {direction} — identity='{identity}' conf={conf:.3f}")

            # Log to database (async)
            threading.Thread(
                target=self._log_crossing,
                args=(tid, cls_name, identity, direction, conf),
                daemon=True,
            ).start()

            # Add to in-memory history
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "tracker_id": tid,
                "class": cls_name,
                "direction": direction,
                "person_name": identity,
                "confidence": round(conf, 3),
            }
            with self._history_lock:
                self.detection_history.insert(0, event)
                if len(self.detection_history) > 200:
                    self.detection_history = self.detection_history[:200]

        # Cleanup old trackers
        stale = [tid for tid in self._track_history if tid not in active_ids]
        for tid in stale:
            # Keep history for a bit in case tracker reappears briefly
            pass  # We rely on deque maxlen and _counted_ids to prevent re-counting

    # -----------------------------------------------------------------
    # DB logging
    # -----------------------------------------------------------------
    def _log_crossing(self, tracker_id, cls_name, identity, direction, confidence):
        """Save crossing event to database."""
        session = get_session()
        try:
            # Attendance log
            log_entry = AttendanceLog(
                camera_channel=self.channel,
                person_name=identity,
                tracker_id=tracker_id,
                direction=direction,
                confidence=confidence,
            )
            session.add(log_entry)
            # Detection event
            session.add(DetectionEvent(
                camera_channel=self.channel,
                tracker_id=tracker_id,
                object_class=cls_name,
                confidence=confidence,
                direction=direction,
                person_name=identity,
            ))
            session.commit()
            # Store log ID so we can update Unknown→name later
            if identity == "Unknown":
                self._logged_tracker_ids[tracker_id] = log_entry.id
            print(f"[DB] Logged {direction} for '{identity}' (tracker #{tracker_id})")
        except Exception as e:
            session.rollback()
            print(f"[DB] Log error: {e}")
        finally:
            session.close()

    def _update_attendance_identity(self, tracker_id, new_name, confidence):
        """Update a previously-logged Unknown attendance record with the real name."""
        log_id = self._logged_tracker_ids.get(tracker_id)
        if not log_id:
            return
        session = get_session()
        try:
            log_entry = session.query(AttendanceLog).get(log_id)
            if log_entry and log_entry.person_name == "Unknown":
                log_entry.person_name = new_name
                log_entry.confidence = confidence
                session.commit()
                del self._logged_tracker_ids[tracker_id]
                print(f"[DB] Updated attendance #{log_id}: Unknown → '{new_name}'")
        except Exception as e:
            session.rollback()
            print(f"[DB] Update error: {e}")
        finally:
            session.close()

    # -----------------------------------------------------------------
    # Reset counts
    # -----------------------------------------------------------------
    def reset_counts(self):
        self.in_count.clear()
        self.out_count.clear()
        self._counted_ids.clear()
        self._track_history.clear()
        self._tracker_identity.clear()
        self._tracker_face_conf.clear()
        self._tracker_face_attempts.clear()
        self._logged_tracker_ids.clear()

    # -----------------------------------------------------------------
    # MJPEG generator
    # -----------------------------------------------------------------
    def generate_mjpeg(self):
        while self.running:
            with self.lock:
                jpeg = self.frame_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.05)


# =============================================================================
# CAMERA MANAGER — singleton
# =============================================================================
streams: dict[str, CameraStream] = {}
_streams_lock = threading.Lock()


def get_or_create_stream(channel: str) -> CameraStream | None:
    with _streams_lock:
        if channel in streams:
            return streams[channel]
    # Fetch from DB
    session = get_session()
    try:
        cam = session.query(CameraConfig).filter_by(channel=channel).first()
        if not cam:
            return None
        s = CameraStream(cam.channel, cam.rtsp_url, cam.name)
        with _streams_lock:
            streams[channel] = s
        return s
    finally:
        session.close()


# =============================================================================
# API ROUTES
# =============================================================================

# ---- Cameras ----

@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    session = get_session()
    try:
        cams = session.query(CameraConfig).all()
        result = []
        for c in cams:
            s = streams.get(c.channel)
            result.append({
                "id": c.id,
                "name": c.name,
                "channel": c.channel,
                "rtsp_url": c.rtsp_url,
                "is_active": c.is_active,
                "connected": s.connected if s else False,
                "running": s.running if s else False,
            })
        return jsonify(result)
    finally:
        session.close()


@app.route("/api/cameras", methods=["POST"])
def api_add_camera():
    data = request.json or {}
    name = data.get("name", "").strip()
    channel = data.get("channel", "").strip()
    rtsp_url = data.get("rtsp_url", "").strip()
    if not all([name, channel, rtsp_url]):
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    session = get_session()
    try:
        existing = session.query(CameraConfig).filter_by(channel=channel).first()
        if existing:
            return jsonify({"status": "error", "message": "Channel exists"}), 409
        cam = CameraConfig(name=name, channel=channel, rtsp_url=rtsp_url)
        session.add(cam)
        session.commit()
        return jsonify({"status": "ok", "channel": channel})
    except Exception as e:
        session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()


@app.route("/api/cameras/<channel>/connect", methods=["POST"])
def api_connect(channel):
    s = get_or_create_stream(channel)
    if not s:
        return jsonify({"status": "error", "message": "Unknown channel"}), 404
    ok = s.connect()
    if ok:
        s.start()
        return jsonify({"status": "ok", "message": f"{channel} connected"})
    return jsonify({"status": "error", "message": "Failed to connect"}), 500


@app.route("/api/cameras/<channel>/disconnect", methods=["POST"])
def api_disconnect(channel):
    s = streams.get(channel)
    if s:
        s.disconnect()
    return jsonify({"status": "ok"})


# ---- Video feed ----

@app.route("/video_feed/<channel>")
def video_feed(channel):
    s = get_or_create_stream(channel)
    if not s or not s.running:
        return Response("Camera not running", status=503)
    return Response(
        s.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---- Counts ----

@app.route("/api/counts/<channel>")
def api_counts(channel):
    s = streams.get(channel)
    if not s:
        return jsonify({"in": {}, "out": {}})
    return jsonify({"in": dict(s.in_count), "out": dict(s.out_count)})


@app.route("/api/counts/<channel>/reset", methods=["POST"])
def api_reset_counts(channel):
    s = streams.get(channel)
    if s:
        s.reset_counts()
    return jsonify({"status": "ok"})


# ---- Stats ----

@app.route("/api/stats")
def api_stats():
    session = get_session()
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        total = session.query(DetectionEvent).filter(
            DetectionEvent.timestamp >= today
        ).count()

        total_in = session.query(AttendanceLog).filter(
            AttendanceLog.timestamp >= today,
            AttendanceLog.direction == "in",
        ).count()

        total_out = session.query(AttendanceLog).filter(
            AttendanceLog.timestamp >= today,
            AttendanceLog.direction == "out",
        ).count()

        connected = sum(1 for s in streams.values() if s.connected)

        return jsonify({
            "total_detections_today": total,
            "total_in_today": total_in,
            "total_out_today": total_out,
            "connected_cameras": connected,
        })
    finally:
        session.close()


@app.route("/api/faces/status")
def api_faces_status():
    """Debug endpoint — shows loaded face count and names in memory."""
    with _faces_lock:
        return jsonify({
            "loaded_faces": len(_known_encodings),
            "names": list(_known_names),
            "ids": list(_known_ids),
        })


@app.route("/api/faces/reload", methods=["POST"])
def api_faces_reload():
    """Force reload known faces from DB into memory."""
    reload_known_faces()
    with _faces_lock:
        return jsonify({
            "status": "ok",
            "loaded_faces": len(_known_encodings),
            "names": list(_known_names),
        })


# ---- Detection History ----

@app.route("/api/history")
def api_history():
    channel = request.args.get("channel", "ch1")
    limit = int(request.args.get("limit", "30"))

    s = streams.get(channel)
    if not s:
        return jsonify([])

    with s._history_lock:
        return jsonify(s.detection_history[:limit])


# =============================================================================
# FACE MANAGEMENT API
# =============================================================================

@app.route("/api/faces", methods=["GET"])
def api_list_faces():
    """List all known faces."""
    session = get_session()
    try:
        faces = session.query(KnownFace).filter_by(is_active=True).all()
        result = []
        for f in faces:
            thumb_b64 = ""
            if f.thumbnail:
                thumb_b64 = base64.b64encode(f.thumbnail).decode("utf-8")
            result.append({
                "id": f.id,
                "name": f.name,
                "created_at": f.created_at.isoformat() if f.created_at else "",
                "thumbnail": thumb_b64,
            })
        return jsonify(result)
    finally:
        session.close()


@app.route("/api/faces", methods=["POST"])
def api_register_face():
    """
    Register a new known face.
    Accepts multipart form:  name (str) + image (file, JPEG/PNG)
    OR JSON:  name (str) + image_base64 (str, base64-encoded JPEG/PNG)
    """
    name = ""
    image_bytes = None

    if request.content_type and "multipart" in request.content_type:
        name = request.form.get("name", "").strip()
        file = request.files.get("image")
        if file:
            image_bytes = file.read()
    else:
        data = request.json or {}
        name = data.get("name", "").strip()
        b64 = data.get("image_base64", "")
        if b64:
            image_bytes = base64.b64decode(b64)

    if not name or not image_bytes:
        return jsonify({"status": "error", "message": "name + image required"}), 400

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find face(s)
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) == 0:
        return jsonify({"status": "error", "message": "No face detected in image"}), 400

    encodings = face_recognition.face_encodings(rgb, locations)
    if len(encodings) == 0:
        return jsonify({"status": "error", "message": "Could not encode face"}), 400

    # Use the first (largest) face
    enc = encodings[0]
    enc_bytes = pickle.dumps(enc)

    # Create thumbnail (160×160)
    top, right, bottom, left = locations[0]
    face_crop = img[top:bottom, left:right]
    thumb = cv2.resize(face_crop, (160, 160))
    _, thumb_buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    thumb_bytes = thumb_buf.tobytes()

    # Save to DB
    session = get_session()
    try:
        kf = KnownFace(
            name=name,
            encoding=enc_bytes,
            thumbnail=thumb_bytes,
            is_active=True,
        )
        session.add(kf)
        session.commit()
        face_id = kf.id
    except Exception as e:
        session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()

    # Refresh in-memory cache
    reload_known_faces()

    return jsonify({"status": "ok", "face_id": face_id, "name": name})


@app.route("/api/faces/<int:face_id>", methods=["DELETE"])
def api_delete_face(face_id):
    session = get_session()
    try:
        face = session.query(KnownFace).get(face_id)
        if not face:
            return jsonify({"status": "error", "message": "Not found"}), 404
        face.is_active = False
        session.commit()
    except Exception as e:
        session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()

    reload_known_faces()
    return jsonify({"status": "ok"})


# =============================================================================
# ATTENDANCE API
# =============================================================================

@app.route("/api/attendance")
def api_attendance():
    """Get attendance log with optional filters."""
    channel = request.args.get("channel", "")
    person = request.args.get("person", "")
    direction = request.args.get("direction", "")
    limit = int(request.args.get("limit", "50"))
    date_str = request.args.get("date", "")  # YYYY-MM-DD

    session = get_session()
    try:
        q = session.query(AttendanceLog).order_by(AttendanceLog.timestamp.desc())
        if channel:
            q = q.filter(AttendanceLog.camera_channel == channel)
        if person:
            q = q.filter(AttendanceLog.person_name.ilike(f"%{person}%"))
        if direction:
            q = q.filter(AttendanceLog.direction == direction)
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                q = q.filter(
                    AttendanceLog.timestamp >= dt,
                    AttendanceLog.timestamp < dt + timedelta(days=1),
                )
            except ValueError:
                pass

        rows = q.limit(limit).all()
        result = []
        for r in rows:
            result.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "camera_channel": r.camera_channel,
                "person_name": r.person_name,
                "tracker_id": r.tracker_id,
                "direction": r.direction,
                "confidence": round(r.confidence, 3) if r.confidence else 0,
            })
        return jsonify(result)
    finally:
        session.close()


@app.route("/api/attendance/summary")
def api_attendance_summary():
    """Summary: who is currently IN (entered but not exited)."""
    session = get_session()
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        logs = (
            session.query(AttendanceLog)
            .filter(AttendanceLog.timestamp >= today)
            .order_by(AttendanceLog.timestamp.asc())
            .all()
        )

        # Track net direction per person
        person_state: dict[str, str] = {}  # person → last direction
        person_times: dict[str, str] = {}  # person → last timestamp

        for log in logs:
            if log.person_name and log.person_name != "Unknown":
                person_state[log.person_name] = log.direction
                person_times[log.person_name] = log.timestamp.isoformat()

        currently_in = []
        currently_out = []
        for name, direction in person_state.items():
            entry = {"name": name, "last_seen": person_times.get(name, "")}
            if direction == "in":
                currently_in.append(entry)
            else:
                currently_out.append(entry)

        return jsonify({
            "currently_in": currently_in,
            "currently_out": currently_out,
            "total_in": len(currently_in),
            "total_out": len(currently_out),
        })
    finally:
        session.close()


# =============================================================================
# STARTUP
# =============================================================================

def auto_connect_cameras():
    """Try to connect all active cameras on startup."""
    session = get_session()
    try:
        cams = session.query(CameraConfig).filter_by(is_active=1).all()
        for cam in cams:
            s = CameraStream(cam.channel, cam.rtsp_url, cam.name)
            with _streams_lock:
                streams[cam.channel] = s
            if s.connect():
                s.start()
                print(f"[STARTUP] {cam.channel} ({cam.name}) → RUNNING ✓")
            else:
                print(f"[STARTUP] {cam.channel} ({cam.name}) → FAILED (will retry on demand)")
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  CCTV Attendance System — Starting")
    print("=" * 60)

    # Initialize DB
    init_db()

    # Load known faces into memory
    reload_known_faces()

    # Auto-connect cameras
    auto_connect_cameras()

    # Start Flask
    print(f"\n[FLASK] http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
