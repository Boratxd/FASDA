import os
import shutil
import psycopg2
import psycopg2.extras
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

def get_conn():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD
    )
    return conn

def init_flags_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS flagged_students (
            id SERIAL PRIMARY KEY,
            flag_code VARCHAR(20) UNIQUE,
            signature_key TEXT NOT NULL,
            student_id VARCHAR(50),
            student_name VARCHAR(200),
            source_pdf TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_flagged_students_signature_pdf
        ON flagged_students(signature_key, source_pdf)
    """)

    conn.commit()
    cur.close()
    conn.close()

def save_uploaded_pdf(temp_pdf_path):
    init_flags_db()

    filename = os.path.basename(temp_pdf_path)
    name, ext = os.path.splitext(filename)

    final_name = filename
    counter = 1

    while os.path.exists(os.path.join(UPLOADS_DIR, final_name)):
        final_name = f"{name}_{counter}{ext}"
        counter += 1

    final_path = os.path.join(UPLOADS_DIR, final_name)
    shutil.copy2(temp_pdf_path, final_path)

    return final_name

def toggle_flag(signature_key, student_id, student_name, source_pdf=""):
    init_flags_db()

    signature_key = str(signature_key).strip()
    student_id = str(student_id).strip()
    student_name = str(student_name).strip()
    source_pdf = str(source_pdf).strip()

    if not signature_key:
        return {
            "success": False,
            "error": "Missing signature_key"
        }

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:

        cur.execute("""
            SELECT id, flag_code
            FROM flagged_students
            WHERE signature_key = %s AND source_pdf = %s
        """, (signature_key, source_pdf))
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                DELETE FROM flagged_students
                WHERE id = %s
            """, (existing["id"],))
            conn.commit()
            cur.close()
            conn.close()

            return {
                "success": True,
                "action": "unflagged",
                "flag_id": existing["flag_code"]
            }

        cur.execute("""
            INSERT INTO flagged_students
            (flag_code, signature_key, student_id, student_name, source_pdf, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            None,
            signature_key,
            student_id,
            student_name,
            source_pdf,
            datetime.now().isoformat()
        ))

        new_id = cur.fetchone()[0]
        flag_code = f"F{new_id:04d}"

        cur.execute("""
            UPDATE flagged_students
            SET flag_code = %s
            WHERE id = %s
        """, (flag_code, new_id))

        conn.commit()
        cur.close()
        conn.close()

        return {
            "success": True,
            "action": "flagged",
            "flag_id": flag_code
        }

    except psycopg2.IntegrityError:
        conn.rollback()
        cur.execute("""
            SELECT flag_code
            FROM flagged_students
            WHERE signature_key = %s AND source_pdf = %s
        """, (signature_key, source_pdf))
        existing = cur.fetchone()

        cur.close()
        conn.close()

        if existing:
            return {
                "success": True,
                "action": "flagged",
                "flag_id": existing["flag_code"]
            }

        return {
            "success": False,
            "error": "Integrity error while toggling flag"
        }

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        return {
            "success": False,
            "error": str(e)
        }

def get_flagged_map():
    init_flags_db()
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute("""
        SELECT signature_key, source_pdf, flag_code
        FROM flagged_students
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    flagged_map = {}
    for r in rows:
        key = f"{r['source_pdf']}::{r['signature_key']}"
        flagged_map[key] = r["flag_code"]

    return flagged_map
