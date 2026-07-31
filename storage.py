import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "events.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_clip TEXT,
    cat TEXT NOT NULL,          -- white | black | unknown | human | pass_through
    is_day INTEGER NOT NULL,    -- 1 = day, 0 = night
    confidence REAL,
    dwell_seconds REAL,
    clip_path TEXT,
    thumbnail_path TEXT
);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_event(
    conn: sqlite3.Connection,
    timestamp: str,
    cat: str,
    is_day: bool,
    confidence: float | None = None,
    dwell_seconds: float | None = None,
    source_clip: str | None = None,
    clip_path: str | None = None,
    thumbnail_path: str | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO events
           (timestamp, source_clip, cat, is_day, confidence, dwell_seconds, clip_path, thumbnail_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            source_clip,
            cat,
            1 if is_day else 0,
            confidence,
            dwell_seconds,
            clip_path,
            thumbnail_path,
        ),
    )
    conn.commit()
    return cur.lastrowid


def clear_events(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM events")
    conn.commit()
