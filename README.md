# FASDA — Forgery Attendance Signature Detection App

**A machine-learning-powered system for detecting forged signatures on university attendance sheets.**

FASDA is a full-stack web application designed to help instructors automatically verify the authenticity of signatures collected during in-class attendance. The system processes scanned attendance PDF uploads, segments each student's signature, runs them through a hybrid ML pipeline, and reports potential forgeries grouped by student ID.

This project was developed as part of the **CNG491 Graduation Project** at **Middle East Technical University, Northern Cyprus Campus (METU NCC)**.

---

## Features

- Automated forgery detection on scanned PDF attendance sheets
- Hybrid scoring: Siamese CNN + HOG feature comparison
- Median-based robust outlier detection (resistant to anchor contamination)
- Single signature verification against a registered student reference
- Annotated output pages with bounding boxes and decision labels
- Per-student result grouping with genuine / forged breakdown
- Saved reports with aggregate statistics and charts
- Admin panel: user approval, ALPHA weight configuration
- Dark / light mode UI
- Terminal-level performance timing for each pipeline stage

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| Frontend | Jinja2 templates, custom CSS |
| ML | TensorFlow / Keras, OpenCV, scikit-image |
| Database | PostgreSQL (AWS RDS) |
| PDF processing | PyMuPDF (fitz) |

---

## System Architecture

```
┌─────────────────────────────────────┐
│            Browser (HTML/CSS/JS)    │
└────────────────┬────────────────────┘
                 │ HTTP
┌────────────────▼────────────────────┐
│         Flask Backend (app.py)      │
│  ┌──────────────┐  ┌─────────────┐  │
│  │ Auth / Admin │  │ Routes      │  │
│  └──────────────┘  └──────┬──────┘  │
│                           │         │
│  ┌────────────────────────▼──────┐  │
│  │       ML Pipeline             │  │
│  │  PDF Segmentation             │  │
│  │  HOG Feature Extraction       │  │
│  │  Siamese CNN Scoring          │  │
│  │  Median Decision              │  │
│  └───────────────────────────────┘  │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         PostgreSQL (AWS RDS)        │
└─────────────────────────────────────┘
```

---

## ML Pipeline — How It Works

### 1. PDF Segmentation
The uploaded attendance PDF is rendered page by page. A grid-detection algorithm locates each signature cell and crops it individually.

### 2. Feature Extraction
Two parallel signals are computed for each signature:

- **HOG (Histogram of Oriented Gradients):** Encodes stroke direction and ink distribution into a ~500-dimensional feature vector.
- **Siamese CNN:** A trained Convolutional Neural Network that outputs a pairwise similarity score between two signature images.

### 3. Median-Based Decision (Anchor-Free)

Traditional anchor-pair methods pick the two most similar signatures as a reference, which fails when two forged signatures happen to resemble each other (anchor contamination).

FASDA uses an element-wise median of all HOG feature vectors in a row as the reference. The median is statistically robust — even if 1–2 signatures are forged, the majority genuine signatures keep the reference stable.

Each signature is then scored by its cosine similarity to the median vector. A z-score threshold (−0.8 std) flags outliers as **FORGED**.

### 4. Hybrid Score
```
hybrid = ALPHA × CNN_score + (1 − ALPHA) × HOG_score
```
`ALPHA` defaults to `0.15` for attendance analysis (emphasising HOG) and is adjustable from the Admin Panel.

---

## User Manual

### Registration & Login

1. Open the app in your browser.
2. Click **Register** and fill in your name, username, and password.
3. An administrator must approve your account before you can log in.
4. Once approved, log in with your username and password.

### Dashboard

The dashboard is the starting point after login. It shows:

- **Verify Signature** — compare a single uploaded signature against a student record.
- **Upload Attendance Sheet** — analyse a full PDF attendance sheet.
- **My Reports** — access previously saved analysis reports.

Admins see additional options: Admin Panel, Benchmark, and Distribution Plots.

### Verify Signature

1. Navigate to **Verify Signature** from the sidebar or dashboard.
2. Enter the **Student ID** of the student whose signature you want to check.
3. Upload the signature image (JPEG or PNG).
4. Click **Run Verification**.
5. The result panel shows:
   - CNN similarity score
   - HOG similarity score
   - Hybrid score
   - Final decision: **GENUINE** or **FORGED**

### Upload Attendance Sheet

1. Navigate to **Attendance** from the sidebar.
2. Select a scanned attendance PDF.
3. A preview of the PDF appears on the right.
4. Click **Upload & Process**.
5. A progress overlay shows each pipeline stage (segmentation → ML → annotation).
6. You are redirected to the results page when processing completes.

### Attendance Results

The results page is split into two columns:

- **Left — Annotated Pages:** Each page of the attendance sheet with coloured bounding boxes:
  - Green = GENUINE
  - Red = FORGED
- **Right — Students by ID:** Collapsible rows per student showing per-signature CNN, HOG, hybrid scores and the final decision.

You can **flag** individual signatures for manual review using the flag button in each row.

Click **Generate Report** to open a report modal with:
- Pie chart (genuine / forged / unknown distribution)
- Bar chart (per-student breakdown)
- Sortable table by forgery rate

Enter a course name and click **Save Report** to store the report permanently.

### Reports

Saved reports are accessible from **My Reports** on the dashboard. Each folder corresponds to a course. Inside a folder you can:

- View aggregate statistics across all reports in the folder.
- Open individual reports with full charts and student tables.
- Print any report using the Print button.

### Profile

Click your name in the sidebar footer to view your profile — name, username, phone, role, and account status.

### Admin Panel (Administrators only)

Accessible via the sidebar under **Admin Panel**.

- **Model Weight Configuration:** Drag the ALPHA slider to adjust the CNN / HOG balance and click Save.
- **User Management:** Approve pending accounts or delete users. Use the tab buttons to filter by status.

### Theme Toggle

Click the ☀️ / 🌙 button in the bottom-right corner to switch between dark and light mode. The preference is saved in the browser.

---

## Installation Manual

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | latest |
| PostgreSQL | 13 or higher (or AWS RDS endpoint) |
| Git | any recent version |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Boratxd/FASDA.git
cd FASDA
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 4 — Place the Model File

Copy the trained Siamese CNN model file into the backend directory:

```
FASDA/
└── backend/
    └── siamese_model.h5      ← place it here
```

The expected path is configured in `backend/config.py` under `MODEL_PATH`.

### Step 5 — Configure the Application

Open `backend/config.py` and update the following values:

```python
DB_HOST     = "your-rds-endpoint-or-localhost"
DB_PORT     = 5432
DB_NAME     = "fasda"
DB_USER     = "your_db_user"
DB_PASSWORD = "your_db_password"

MODEL_PATH  = "siamese_model.h5"
THRESHOLD   = 0.5      # hybrid score cutoff for /verify
ALPHA       = 0.15     # CNN weight in hybrid score
```

### Step 6 — Set Up the Database

Connect to your PostgreSQL instance and run the provided SQL file:

```bash
psql -h <host> -U <user> -d <dbname> -f backend/setup_db.sql
```

This creates all required tables (users, attendance records, reports, flags).

### Step 7 — Run the Application

```bash
cd backend
python main.py
```

The app starts on `http://127.0.0.1:5000` by default.

Open a browser and navigate to `http://localhost:5000` to access FASDA.

---

## Deployment Guide

This section explains how a developer can deploy FASDA on a server using the source code.

### Requirements

- A Linux server (Ubuntu 20.04+ recommended) or any cloud VM
- Python 3.10+
- PostgreSQL database (local or AWS RDS)
- Gunicorn (WSGI server for production)
- Nginx (optional, recommended as a reverse proxy)

### Step 1 — Server Setup

```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip git -y
```

### Step 2 — Clone and Install

```bash
git clone https://github.com/Boratxd/FASDA.git
cd FASDA
python3.10 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install gunicorn
```

### Step 3 — Configure

Edit `backend/config.py` with your production database credentials and model path (see Installation Manual Step 5).

Place `siamese_model.h5` in the `backend/` directory.

### Step 4 — Database

Run the setup script against your PostgreSQL instance:

```bash
psql -h <rds-endpoint> -U <user> -d fasda -f backend/setup_db.sql
```

### Step 5 — Run with Gunicorn

```bash
cd backend
gunicorn -w 2 -b 0.0.0.0:5000 "app:app"
```

For a background service, create a systemd unit file at `/etc/systemd/system/fasda.service`:

```ini
[Unit]
Description=FASDA Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/FASDA/backend
ExecStart=/home/ubuntu/FASDA/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable fasda
sudo systemctl start fasda
```

### Step 6 — Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Project Directory Structure

```
FASDA/
├── backend/
│   ├── app.py                  # Flask application & routes
│   ├── config.py               # DB credentials, model path, thresholds
│   ├── main.py                 # Entry point
│   ├── setup_db.sql            # Database schema
│   ├── siamese_model.h5        # Trained model (not in repo — add manually)
│   ├── ml/
│   │   ├── analyzer.py         # Main ML pipeline
│   │   ├── cnn_model.py        # Siamese CNN loader
│   │   ├── hog_features.py     # HOG feature extraction
│   │   ├── verification.py     # Single-signature verify
│   │   ├── evaluation.py       # Benchmark utilities
│   │   └── perf.py             # Performance timing
│   └── static/
│       ├── fasda.css           # Design system
│       ├── theme.css           # Light mode overrides
│       ├── pages.css           # Page-specific styles
│       └── theme.js            # Theme toggle + sidebar JS
└── frontend/
    └── templates/              # Jinja2 HTML templates
        ├── login.html
        ├── register.html
        ├── dashboard.html
        ├── verify.html
        ├── attendance_upload.html
        ├── attendance_result.html
        ├── report_list.html
        ├── report_view.html
        ├── admin.html
        ├── profile.html
        └── plots.html
```

---

## Source Code

GitHub Repository: [https://github.com/Boratxd/FASDA](https://github.com/Boratxd/FASDA)

Branch `v0.3.1-alpha` contains the latest stable version of the project.

---

## License

This project was developed for academic purposes as part of CNG491 at METU NCC.
