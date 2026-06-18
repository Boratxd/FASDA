import psycopg2
from datetime import datetime
from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            student_id VARCHAR(50),
            test_image TEXT,
            label VARCHAR(20),
            cnn_similarity DOUBLE PRECISION,
            hog_similarity DOUBLE PRECISION,
            hybrid_score DOUBLE PRECISION
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def log_result(student_id, test_image, label, cnn, hog, hybrid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO verification_logs
        (timestamp, student_id, test_image, label,
         cnn_similarity, hog_similarity, hybrid_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (datetime.now().isoformat(), student_id, test_image, label, float(cnn), float(hog), float(hybrid)))
    conn.commit()
    cur.close()
    conn.close()
