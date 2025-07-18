import sqlite3

DB_NAME = "database/attendance_system.db"

def create_database():
    """
    Creates the SQLite database with tables for:
    - students: basic info and face embeddings
    - timetable: lecture slot times
    - attendance: daily attendance entries
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll_no TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timetable (
                slot_id INTEGER PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                roll_no TEXT NOT NULL,
                date TEXT NOT NULL,
                slot_id INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                status TEXT NOT NULL,
                PRIMARY KEY (roll_no, date, slot_id),
                FOREIGN KEY (roll_no) REFERENCES students(roll_no),
                FOREIGN KEY (slot_id) REFERENCES timetable(slot_id)
            )
        """)

def insert_timetable():
    """
    Inserts predefined time slots into the timetable table.
    """
    slots = [
        (1, "08:50", "09:40"),
        (2, "09:40", "10:30"),
        (3, "10:30", "10:45"),
        (4, "10:45", "11:35"),
        (5, "11:35", "12:25"),
        (6, "12:25", "13:15"),
        (7, "13:15", "14:05"),
        (8, "14:05", "14:55"),
        (9, "14:55", "15:45"),
        (10, "15:45", "16:35")
    ]

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO timetable (slot_id, start_time, end_time) VALUES (?, ?, ?)", 
            slots
        )

if __name__ == "__main__":
    create_database()
    insert_timetable()
    print("✅ Database setup complete.")
