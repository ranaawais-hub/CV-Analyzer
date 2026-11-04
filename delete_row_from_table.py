import sqlite3

from cv_analyzer_table_db import DB_FILE   # same database file the main script uses

conn = sqlite3.connect(DB_FILE)
rows = conn.execute(
    "DELETE FROM records WHERE id = ?",
    (4,)
)

conn.commit()
conn.close()