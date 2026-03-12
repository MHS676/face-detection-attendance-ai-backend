"""Quick test to find the working RTSP URL for Camera 2."""
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

urls = [
    "rtsp://admin:Admin0123@192.168.0.50:8554/",
    "rtsp://admin:Admin0123@192.168.0.50:8554/stream1",
    "rtsp://admin:Admin0123@192.168.0.50:8554/live",
    "rtsp://admin:Admin0123@192.168.0.50:8554/ch01/0",
    "rtsp://192.168.0.50:8554/",
    "rtsp://admin:Admin0123@192.168.0.50:8554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.50:8554/",
    "rtsp://admin:Admin321@192.168.0.50:8554/",
    "rtsp://admin:Admin0123@192.168.0.50:8554/h264",
    "rtsp://admin:Admin0123@192.168.0.50:8554/cam/realmonitor?channel=1&subtype=0",
]

working = None

for url in urls:
    print(f"Trying: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ SUCCESS! Frame: {frame.shape}")
            working = url
            cap.release()
            break
        else:
            print("  ⚠️ Opened but no frames")
    else:
        print("  ❌ Failed to open")
    cap.release()

if not working:
    print("\nTrying default backend (no FFMPEG)...")
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✅ SUCCESS! Frame: {frame.shape}")
                working = url
                cap.release()
                break
            else:
                print("  ⚠️ Opened but no frames")
        else:
            print("  ❌ Failed to open")
        cap.release()

if working:
    print(f"\n🎉 Working URL: {working}")
else:
    print("\n❌ All URLs failed for Camera 2")
