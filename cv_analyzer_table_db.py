"""
Database code for the CV screening script.

Put this file in the same folder as screen_cvs.py.
screen_cvs.py imports init_db() and save_to_db() from here.
"""

import sqlite3
from pathlib import Path

# the database file is always created next to this script, wherever you run it from
DB_FILE = str(Path(__file__).parent / "analyzer.db")


def init_db():
    """Creates the table if it does not exist yet."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            file           TEXT NOT NULL UNIQUE,
            name           TEXT,
            email          TEXT,
            years          REAL,
            skills_matched INTEGER NOT NULL,
            status         TEXT NOT NULL,
            reason         TEXT,
            checked_by     TEXT,
            summary        TEXT,
            analyzed_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(results):
    """Saves one row per CV. Running again on the same file updates its row."""
    conn = sqlite3.connect(DB_FILE)
    for r in results:
        conn.execute("""
            INSERT INTO records
                (file, name, email, years, skills_matched, status, reason, checked_by, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                years = excluded.years,
                skills_matched = excluded.skills_matched,
                status = excluded.status,
                reason = excluded.reason,
                checked_by = excluded.checked_by,
                summary = excluded.summary,
                analyzed_at = CURRENT_TIMESTAMP
        """, (
            r["file"],
            r["name"],
            r["email"],
            None if r["years"] == "" else float(r["years"]),
            r["skills_matched"],
            r["final_status"],
            r["reason"],
            r["checked_by"],
            r["summary"],
        ))
    conn.commit()
    conn.close()