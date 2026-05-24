import sqlite3
import string

DB_FILE = "database.db"

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS characters")

cur.execute("""
CREATE TABLE characters (
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (name, status)
)
"""
)

data = [
    ("teacher", "real"),
    ("teacher", "portrait"),
    ("studentMain", "real"),
    ("studentMain", "portrait"),
]


for name_label in string.ascii_uppercase[:17]:
    for status in ["real","portrait"]:
        data.append(("student"+name_label,status))

cur.executemany(
    "INSERT INTO characters (name, status) VALUES (?, ?)",
    data
)

conn.commit()


rows = cur.fetchall()

conn.close()