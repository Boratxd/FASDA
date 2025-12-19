# ml/database.py
import os
import sqlite3
from datetime import datetime
from config import DB_PATH

# =========================================================
# SQLITE INITIALIZATION
# =========================================================
def init_db():
    # Ensure folder exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS verification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            student_id TEXT,
            test_image TEXT,
            label TEXT,
            cnn_similarity REAL,
            hog_similarity REAL,
            hybrid_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_result(student_id, test_image, label, cnn, hog, hybrid):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO verification_logs
        (timestamp, student_id, test_image, label,
         cnn_similarity, hog_similarity, hybrid_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        student_id,
        test_image,
        label,
        float(cnn),
        float(hog),
        float(hybrid)
    ))
    conn.commit()
    conn.close()
