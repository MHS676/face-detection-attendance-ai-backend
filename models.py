"""
Database models for CCTV Attendance + Face Recognition System.
Uses PostgreSQL via SQLAlchemy.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Index, LargeBinary, Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set. Add it to your .env file.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# =============================================================================
# MODELS
# =============================================================================

class KnownFace(Base):
    """Stores known face encodings for identification."""
    __tablename__ = "known_faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    encoding = Column(LargeBinary, nullable=False)       # 128-d numpy array → bytes
    thumbnail = Column(LargeBinary, nullable=True)       # JPEG thumbnail of face
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    __table_args__ = (
        Index("idx_known_face_name", "name"),
    )


class AttendanceLog(Base):
    """
    Main attendance record — logs every identified person in/out event.
    """
    __tablename__ = "attendance_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    camera_channel = Column(String(50), nullable=False, index=True)
    person_name = Column(String(200), nullable=False, index=True)
    face_id = Column(Integer, nullable=True)
    tracker_id = Column(Integer, nullable=False)
    direction = Column(String(10), nullable=True)         # "in" / "out"
    confidence = Column(Float, nullable=True)             # face match distance
    thumbnail = Column(LargeBinary, nullable=True)        # face crop JPEG

    __table_args__ = (
        Index("idx_attend_person_time", "person_name", "timestamp"),
        Index("idx_attend_cam_time", "camera_channel", "timestamp"),
    )


class DetectionEvent(Base):
    """Stores detection events for history."""
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    camera_channel = Column(String(50), nullable=False, index=True)
    tracker_id = Column(Integer, nullable=False)
    object_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    direction = Column(String(10), nullable=True)
    person_name = Column(String(200), nullable=True)

    __table_args__ = (
        Index("idx_det_camera_time", "camera_channel", "timestamp"),
    )


class ObjectCount(Base):
    """Aggregated counts per camera per direction."""
    __tablename__ = "object_counts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    camera_channel = Column(String(50), nullable=False, index=True)
    object_class = Column(String(100), nullable=False)
    direction = Column(String(10), nullable=False)
    count = Column(Integer, default=0)
    person_name = Column(String(200), nullable=True)

    __table_args__ = (
        Index("idx_count_camera_dir", "camera_channel", "direction"),
    )


class CameraConfig(Base):
    """Camera/stream configurations."""
    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    channel = Column(String(50), nullable=False, unique=True)
    rtsp_url = Column(Text, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db():
    """Create all tables and seed defaults."""
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created/verified successfully.")

    session = SessionLocal()
    try:
        if session.query(CameraConfig).count() == 0:
            default_cameras = [
                CameraConfig(
                    name="Camera 1 - Main",
                    channel="ch1",
                    rtsp_url=os.getenv("DEFAULT_CAM1_URL", ""),
                    is_active=1,
                ),
                CameraConfig(
                    name="Camera 2 - Alt",
                    channel="ch2",
                    rtsp_url=os.getenv("DEFAULT_CAM2_URL", ""),
                    is_active=1,
                ),
            ]
            session.add_all(default_cameras)
            session.commit()
            print("[DB] Default camera configs seeded.")
    except Exception as e:
        session.rollback()
        print(f"[DB] Seed warning: {e}")
    finally:
        session.close()


def get_session():
    return SessionLocal()


if __name__ == "__main__":
    init_db()
    print("[DB] Database initialized successfully!")
