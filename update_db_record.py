import sqlite3

from cv_analyzer_table_db import DB_FILE   # same database file the main script uses

conn = sqlite3.connect(DB_FILE)
rows = conn.execute(
    "UPDATE records SET status = ? WHERE id = ?",
    ("SHORTLISTED", 2)
)

conn.commit()
conn.close()