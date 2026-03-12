"""Quick test for channels 4-16 on the DVR at 192.168.0.29:554"""
import cv2

for ch in range(4, 17):
    url = f"rtsp://admin:Admin0123@192.168.0.29:554/cam/realmonitor?channel={ch}&subtype=0"
    print(f"  Ch{ch}: {url} ...", end=" ", flush=True)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✅ ({w}×{h})")
        else:
            print("❌ opened but no frame")
        cap.release()
    else:
        print("❌")
        cap.release()
        # Once we start getting failures, the rest are likely empty too
        # But let's test a few more

print("\nDone!")
