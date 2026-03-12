"""
Update camera configs in DB to use Dahua DVR channels.
DVR at 192.168.0.29:554 has 16 channels.
"""

import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from models import init_db, get_session, CameraConfig

_dvr_user = os.getenv("DVR_USER", "admin")
_dvr_pass = os.getenv("DVR_PASS", "")
_dvr_host = os.getenv("DVR_HOST", "192.168.0.29")
_dvr_port = os.getenv("DVR_PORT", "554")
DVR_BASE = f"rtsp://{_dvr_user}:{_dvr_pass}@{_dvr_host}:{_dvr_port}/cam/realmonitor"

# All 16 channels — ch1-ch3 active, ch4-ch16 inactive (user can start from UI)
CHANNELS = [
    {"channel": "ch1",  "name": "DVR Ch1 - Camera 1",   "active": 1},
    {"channel": "ch2",  "name": "DVR Ch2 - Camera 2",   "active": 1},
    {"channel": "ch3",  "name": "DVR Ch3 - Camera 3",   "active": 1},
    {"channel": "ch4",  "name": "DVR Ch4 - Camera 4",   "active": 0},
    {"channel": "ch5",  "name": "DVR Ch5 - Camera 5",   "active": 0},
    {"channel": "ch6",  "name": "DVR Ch6 - Camera 6",   "active": 0},
    {"channel": "ch7",  "name": "DVR Ch7 - Camera 7",   "active": 0},
    {"channel": "ch8",  "name": "DVR Ch8 - Camera 8",   "active": 0},
    {"channel": "ch9",  "name": "DVR Ch9 - Camera 9",   "active": 0},
    {"channel": "ch10", "name": "DVR Ch10 - Camera 10",  "active": 0},
    {"channel": "ch11", "name": "DVR Ch11 - Camera 11",  "active": 0},
    {"channel": "ch12", "name": "DVR Ch12 - Camera 12",  "active": 0},
    {"channel": "ch13", "name": "DVR Ch13 - Camera 13",  "active": 0},
    {"channel": "ch14", "name": "DVR Ch14 - Camera 14",  "active": 0},
    {"channel": "ch15", "name": "DVR Ch15 - Camera 15",  "active": 0},
    {"channel": "ch16", "name": "DVR Ch16 - Camera 16",  "active": 0},
]

def make_url(ch_num: int) -> str:
    return f"{DVR_BASE}?channel={ch_num}&subtype=0"


def main():
    init_db()
    session = get_session()
    try:
        for i, ch in enumerate(CHANNELS, start=1):
            url = make_url(i)
            existing = session.query(CameraConfig).filter_by(channel=ch["channel"]).first()
            if existing:
                old_url = existing.rtsp_url
                existing.rtsp_url = url
                existing.name = ch["name"]
                existing.is_active = ch["active"]
                print(f"  ✏️  {ch['channel']} updated: {old_url} → {url}  (active={ch['active']})")
            else:
                cam = CameraConfig(
                    name=ch["name"],
                    channel=ch["channel"],
                    rtsp_url=url,
                    is_active=ch["active"],
                )
                session.add(cam)
                print(f"  ➕  {ch['channel']} added: {url}  (active={ch['active']})")

        session.commit()
        print(f"\n✅  All {len(CHANNELS)} camera configs written to database.")

        # Verify
        print("\n--- Current camera_configs ---")
        for cam in session.query(CameraConfig).order_by(CameraConfig.channel).all():
            status = "🟢 active" if cam.is_active else "⚪ inactive"
            print(f"  {cam.channel:6s}  {cam.name:25s}  {status}  {cam.rtsp_url}")

    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
