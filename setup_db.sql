-- 1. verification_logs tablosu (mevcut SQLite'dan aktarim)
CREATE TABLE IF NOT EXISTS verification_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    student_id VARCHAR(50),
    test_image TEXT,
    label VARCHAR(20),
    cnn_similarity DOUBLE PRECISION,
    hog_similarity DOUBLE PRECISION,
    hybrid_score DOUBLE PRECISION
);

-- 2. users tablosu
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    payment_status VARCHAR(20) DEFAULT 'unpaid',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. flagged_students tablosu
CREATE TABLE IF NOT EXISTS flagged_students (
    id SERIAL PRIMARY KEY,
    flag_code VARCHAR(20) UNIQUE,
    signature_key TEXT NOT NULL,
    student_id VARCHAR(50),
    student_name VARCHAR(200),
    row_index INTEGER,
    page_number INTEGER,
    note TEXT,
    source_pdf TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_flagged_students_signature_pdf
ON flagged_students(signature_key, source_pdf);

-- 4. reports tablosu
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    report_name VARCHAR(255) NOT NULL,
    report_type VARCHAR(50),
    source_pdf TEXT,
    generated_by VARCHAR(100),
    summary TEXT,
    file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
