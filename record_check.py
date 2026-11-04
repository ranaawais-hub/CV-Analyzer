import sqlite3

from cv_analyzer_table_db import DB_FILE   # same database file the main script uses

conn = sqlite3.connect(DB_FILE)
rows = conn.execute(
    "SELECT id, email, skills_matched, status, analyzed_at, file FROM records"
).fetchall()

for r in rows:
    print(r)

print("\nTotal rows:", len(rows))
conn.close()