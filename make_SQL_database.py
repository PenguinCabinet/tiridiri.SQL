import sqlite3
import string

def make_all_characters():
    data = [
        ("teacher", "real"),
        ("teacher", "portrait"),
        ("studentMain", "real"),
        ("studentMain", "portrait"),
    ]


    for name_label in string.ascii_uppercase[:17]:
        for status in ["real","portrait"]:
            data.append(("student"+name_label,status))
    return data

if __name__=="__main__":

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

    characters = make_all_characters()

    cur.executemany(
        "INSERT INTO characters (name, status) VALUES (?, ?)",
        characters
    )

    conn.commit()


    rows = cur.fetchall()

    conn.close()