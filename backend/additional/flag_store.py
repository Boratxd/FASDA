import os
import shutil
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "flags.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_flags_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS flagged_students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flag_code TEXT UNIQUE,
            signature_key TEXT NOT NULL,
            student_id TEXT,
            student_name TEXT,
            row_index INTEGER,
            page_number INTEGER,
            note TEXT,
            source_pdf TEXT,
            created_at TEXT
        )
    """)

    conn.commit()

    # Old databases may still have UNIQUE(signature_key) from previous schema.
    # This migration recreates the table without forcing signature_key to be unique globally.
    cur.execute("PRAGMA table_info(flagged_students)")
    columns = cur.fetchall()

    signature_key_unique = False
    for col in columns:
        # pragma table_info does not directly expose unique info,
        # so we also inspect indexes below.
        pass

    cur.execute("PRAGMA index_list(flagged_students)")
    indexes = cur.fetchall()

    unique_signature_key_index = None
    for idx in indexes:
        # idx fields: seq, name, unique, origin, partial
        if idx["unique"] == 1:
            idx_name = idx["name"]
            cur.execute(f"PRAGMA index_info({idx_name})")
            idx_cols = cur.fetchall()
            idx_col_names = [c["name"] for c in idx_cols]
            if idx_col_names == ["signature_key"]:
                unique_signature_key_index = idx_name
                break

    if unique_signature_key_index:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS flagged_students_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flag_code TEXT UNIQUE,
                signature_key TEXT NOT NULL,
                student_id TEXT,
                student_name TEXT,
                row_index INTEGER,
                page_number INTEGER,
                note TEXT,
                source_pdf TEXT,
                created_at TEXT
            )
        """)

        cur.execute("""
            INSERT INTO flagged_students_new
            (id, flag_code, signature_key, student_id, student_name, row_index, page_number, note, source_pdf, created_at)
            SELECT id, flag_code, signature_key, student_id, student_name, row_index, page_number, note, source_pdf, created_at
            FROM flagged_students
        """)

        cur.execute("DROP TABLE flagged_students")
        cur.execute("ALTER TABLE flagged_students_new RENAME TO flagged_students")
        conn.commit()

    # Composite uniqueness for same signature in same PDF only
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_flagged_students_signature_pdf
        ON flagged_students(signature_key, source_pdf)
    """)

    conn.commit()
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


def _normalize_row_index(row_index):
    try:
        return int(row_index) if row_index is not None and str(row_index).strip() != "" else None
    except Exception:
        return None


def _normalize_page_number(page_number):
    try:
        return int(page_number) if page_number is not None and str(page_number).strip() != "" else None
    except Exception:
        return None


def toggle_flag(signature_key, student_id, student_name, row_index, page_number, note="", source_pdf=""):
    init_flags_db()

    signature_key = str(signature_key).strip()
    student_id = str(student_id).strip()
    student_name = str(student_name).strip()
    note = str(note).strip()
    source_pdf = str(source_pdf).strip()
    row_index = _normalize_row_index(row_index)
    page_number = _normalize_page_number(page_number)

    if not signature_key:
        return {
            "success": False,
            "error": "Missing signature_key"
        }

    conn = get_conn()
    cur = conn.cursor()

    try:
        # If already flagged in same PDF -> unflag
        cur.execute("""
            SELECT id, flag_code
            FROM flagged_students
            WHERE signature_key = ? AND source_pdf = ?
        """, (signature_key, source_pdf))
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                DELETE FROM flagged_students
                WHERE id = ?
            """, (existing["id"],))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "action": "unflagged",
                "flag_id": existing["flag_code"]
            }

        # Insert new flag
        cur.execute("""
            INSERT INTO flagged_students
            (flag_code, signature_key, student_id, student_name, row_index, page_number, note, source_pdf, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            None,
            signature_key,
            student_id,
            student_name,
            row_index,
            page_number,
            note,
            source_pdf,
            datetime.now().isoformat()
        ))

        new_id = cur.lastrowid
        flag_code = f"F{new_id:04d}"

        cur.execute("""
            UPDATE flagged_students
            SET flag_code = ?
            WHERE id = ?
        """, (flag_code, new_id))

        conn.commit()
        conn.close()

        return {
            "success": True,
            "action": "flagged",
            "flag_id": flag_code
        }

    except sqlite3.IntegrityError:
        # If another request inserted same row almost simultaneously
        cur.execute("""
            SELECT flag_code
            FROM flagged_students
            WHERE signature_key = ? AND source_pdf = ?
        """, (signature_key, source_pdf))
        existing = cur.fetchone()

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
        conn.close()
        return {
            "success": False,
            "error": str(e)
        }


def get_flagged_map():
    init_flags_db()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT signature_key, source_pdf, flag_code
        FROM flagged_students
    """)
    rows = cur.fetchall()
    conn.close()

    flagged_map = {}
    for r in rows:
        key = f"{r['source_pdf']}::{r['signature_key']}"
        flagged_map[key] = r["flag_code"]

    return flagged_map
