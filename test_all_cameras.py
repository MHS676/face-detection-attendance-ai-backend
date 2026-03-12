"""
Comprehensive RTSP URL tester for all cameras.
Tests multiple URL patterns for Camera 2 (192.168.0.50:8554)
and checks Camera 1 for additional channels/streams.
"""
import cv2
import sys
import time

def test_url(url, timeout=8):
    """Try to open an RTSP URL and read one frame."""
    print(f"  Testing: {url} ...", end=" ", flush=True)
    
    # Try FFMPEG backend first
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            cap.release()
            print(f"✅ SUCCESS ({w}×{h})")
            return True
        cap.release()
    
    # Try default backend
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            cap.release()
            print(f"✅ SUCCESS - default backend ({w}×{h})")
            return True
        cap.release()
    
    print("❌ FAILED")
    return False


def main():
    results = {}
    
    # ============================================
    # Camera 1 - 192.168.0.29:554 (known working)
    # Test additional sub-streams / channels
    # ============================================
    print("\n" + "="*60)
    print("CAMERA 1 — 192.168.0.29:554")
    print("="*60)
    
    cam1_urls = [
        # Main stream (known working)
        "rtsp://admin:Admin0123@192.168.0.29:554/",
        # Hikvision patterns
        "rtsp://admin:Admin0123@192.168.0.29:554/Streaming/Channels/101",
        "rtsp://admin:Admin0123@192.168.0.29:554/Streaming/Channels/102",
        "rtsp://admin:Admin0123@192.168.0.29:554/Streaming/Channels/201",
        "rtsp://admin:Admin0123@192.168.0.29:554/Streaming/Channels/301",
        # Dahua patterns
        "rtsp://admin:Admin0123@192.168.0.29:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:Admin0123@192.168.0.29:554/cam/realmonitor?channel=1&subtype=1",
        "rtsp://admin:Admin0123@192.168.0.29:554/cam/realmonitor?channel=2&subtype=0",
        "rtsp://admin:Admin0123@192.168.0.29:554/cam/realmonitor?channel=3&subtype=0",
        # Generic patterns
        "rtsp://admin:Admin0123@192.168.0.29:554/stream1",
        "rtsp://admin:Admin0123@192.168.0.29:554/stream2",
        "rtsp://admin:Admin0123@192.168.0.29:554/ch01/0",
        "rtsp://admin:Admin0123@192.168.0.29:554/ch01/1",
        "rtsp://admin:Admin0123@192.168.0.29:554/ch02/0",
        "rtsp://admin:Admin0123@192.168.0.29:554/ch03/0",
        "rtsp://admin:Admin0123@192.168.0.29:554/live",
        "rtsp://admin:Admin0123@192.168.0.29:554/h264",
        "rtsp://admin:Admin0123@192.168.0.29:554/media/video1",
        "rtsp://admin:Admin0123@192.168.0.29:554/media/video2",
        # ONVIF patterns
        "rtsp://admin:Admin0123@192.168.0.29:554/onvif1",
        "rtsp://admin:Admin0123@192.168.0.29:554/onvif2",
        # Profile patterns
        "rtsp://admin:Admin0123@192.168.0.29:554/profile1",
        "rtsp://admin:Admin0123@192.168.0.29:554/profile2",
        "rtsp://admin:Admin0123@192.168.0.29:554/profile3",
    ]
    
    for url in cam1_urls:
        ok = test_url(url, timeout=5)
        results[url] = ok
    
    # ============================================
    # Camera 2 — 192.168.0.50:8554
    # ============================================
    print("\n" + "="*60)
    print("CAMERA 2 — 192.168.0.50:8554")
    print("="*60)
    
    # Try with different credentials too
    creds = [
        "admin:Admin0123",
        "admin:admin",
        "admin:12345",
        "admin:123456",
        "admin:",
        "",  # no auth
    ]
    
    paths = [
        "/",
        "/stream1",
        "/stream2",
        "/live",
        "/live/ch00_0",
        "/live/ch01_0",
        "/h264",
        "/h264Preview_01_main",
        "/h264Preview_01_sub",
        "/cam/realmonitor?channel=1&subtype=0",
        "/cam/realmonitor?channel=1&subtype=1",
        "/Streaming/Channels/101",
        "/Streaming/Channels/102",
        "/ch01/0",
        "/ch01/1",
        "/media/video1",
        "/onvif1",
        "/profile1",
        "/video",
        "/video1",
        "/0",
        "/1",
        "/ch0_0.h264",
        "/user=admin&password=Admin0123&channel=1&stream=0.sdp",
    ]
    
    for cred in creds:
        print(f"\n--- Credentials: '{cred}' ---")
        for path in paths:
            if cred:
                url = f"rtsp://{cred}@192.168.0.50:8554{path}"
            else:
                url = f"rtsp://192.168.0.50:8554{path}"
            ok = test_url(url, timeout=5)
            results[url] = ok
            if ok:
                print(f"\n🎉🎉🎉 FOUND WORKING URL: {url} 🎉🎉🎉\n")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY — Working URLs")
    print("="*60)
    working = {u: r for u, r in results.items() if r}
    if working:
        for url in working:
            print(f"  ✅ {url}")
    else:
        print("  No working URLs found for Camera 2")
    
    print(f"\nTotal tested: {len(results)}")
    print(f"Working: {len(working)}")


if __name__ == "__main__":
    main()
