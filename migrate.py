"""
Database migration: add new columns/tables for face recognition.
Run once to update existing database schema.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        # Add person_name to detection_events if missing
        try:
            conn.execute(text("ALTER TABLE detection_events ADD COLUMN person_name VARCHAR(200)"))
            conn.commit()
            print("Added person_name to detection_events")
        except Exception as e:
            conn.rollback()
            print(f"detection_events.person_name: already exists or {e}")

        # Add person_name to object_counts if missing
        try:
            conn.execute(text("ALTER TABLE object_counts ADD COLUMN person_name VARCHAR(200)"))
            conn.commit()
            print("Added person_name to object_counts")
        except Exception as e:
            conn.rollback()
            print(f"object_counts.person_name: already exists or {e}")

        # Create known_faces table
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS known_faces (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    encoding BYTEA NOT NULL,
                    thumbnail BYTEA,
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                )
            """))
            conn.commit()
            print("known_faces table OK")
        except Exception as e:
            conn.rollback()
            print(f"known_faces: {e}")

        # Create attendance_log table
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS attendance_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW() NOT NULL,
                    camera_channel VARCHAR(50) NOT NULL,
                    person_name VARCHAR(200) NOT NULL,
                    face_id INTEGER,
                    tracker_id INTEGER NOT NULL,
                    direction VARCHAR(10),
                    confidence FLOAT,
                    thumbnail BYTEA
                )
            """))
            conn.commit()
            print("attendance_log table OK")
        except Exception as e:
            conn.rollback()
            print(f"attendance_log: {e}")

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_known_face_name ON known_faces (name)",
            "CREATE INDEX IF NOT EXISTS idx_attend_person_time ON attendance_log (person_name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_attend_cam_time ON attendance_log (camera_channel, timestamp)",
        ]
        for sql in indexes:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Index: {e}")

        print("Migration complete!")


if __name__ == "__main__":
    migrate()
