import os
import re
import json
import time
import psycopg2
import psycopg2.extras
import tensorflow as tf
from collections import defaultdict
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.utils import secure_filename

import config as _app_config
from additional.attendanceSegmantationPort import process_attendance_pdf
from additional.flag_store import toggle_flag, get_flagged_map, save_uploaded_pdf
from ml.verification import verify_signature
from ml.evaluation import plot_similarity_distributions, evaluate_distributions
from ml.custom_layers import L1DistanceLayer
from ml.analyzer import analyze_and_annotate
from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

VERIFY_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ATTENDANCE_UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "attendance", "uploads")
RENDERED_PAGES_FOLDER = os.path.join(BASE_DIR, "static", "attendance", "rendered_pages")
REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gpds_siamese_modelv4.h5")
DATASET_PATH = r"C:\Users\alper\Desktop\CNG 491\DataSet\SignatureGPDSSyntheticSignaturesManuscriptsv\firmasSINTESISmanuscritas"

os.makedirs(VERIFY_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RENDERED_PAGES_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)

app.secret_key = "fasda_demo_secret"
app.config["UPLOAD_FOLDER"] = VERIFY_UPLOAD_FOLDER

def get_db():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"L1DistanceLayer": L1DistanceLayer}
)

def _run_db_migration():
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS approved BOOLEAN DEFAULT FALSE"
        )
        cur.execute("UPDATE users SET approved = TRUE WHERE role = 'admin'")

        cur.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'reports' AND column_name = 'report_name'
        """)
        has_old_schema = cur.fetchone()[0] > 0
        if has_old_schema:
            cur.execute("DROP TABLE reports CASCADE")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id         SERIAL PRIMARY KEY,
                username   VARCHAR(100) NOT NULL DEFAULT '',
                course     VARCHAR(200),
                slug       VARCHAR(100),
                source_pdf VARCHAR(300),
                summary    JSONB,
                results    JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reports_username_slug ON reports (username, slug)")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[FASDA] DB migration warning: {exc}")

_run_db_migration()

def normalize_page_paths(raw_pages):
    if isinstance(raw_pages, dict):
        try:
            iterable = [raw_pages[k] for k in sorted(raw_pages.keys(), key=lambda x: int(x))]
        except Exception:
            iterable = list(raw_pages.values())
    elif isinstance(raw_pages, list):
        iterable = raw_pages
    else:
        iterable = []

    normalized_pages = []
    for item in iterable:
        if isinstance(item, dict):
            page_path = item.get("image_path") or item.get("page_path") or item.get("path") or ""
        else:
            page_path = str(item)

        if not page_path:
            continue

        page_path = page_path.replace("\\", "/")

        if "/static/" in page_path:
            page_path = "/static/" + page_path.split("/static/", 1)[1]
        elif "static/" in page_path:
            page_path = "/static/" + page_path.split("static/", 1)[1]
        elif not page_path.startswith("/static/"):
            filename = os.path.basename(page_path)
            page_path = f"/static/attendance/rendered_pages/{filename}"

        normalized_pages.append(page_path)

    return normalized_pages

def group_results_by_student(results):
    groups = defaultdict(list)
    for r in results:
        sid = r.get("student_id") or "Unknown"
        if str(sid).strip() in ("", "None", "null"):
            sid = "Unknown"
        groups[str(sid)].append(r)
    return dict(sorted(groups.items()))

def safe_json_results(results):
    def _convert(v):
        if v is None:
            return None
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    serialized = []
    for r in results:
        if hasattr(r, "items"):
            serialized.append({k: _convert(v) for k, v in r.items()})
        else:
            serialized.append(str(r))
    try:
        return json.dumps(serialized)
    except Exception:
        return "[]"

def get_display_name():
    first = session.get("first_name", "")
    last = session.get("last_name", "")
    name = f"{first} {last}".strip()
    return name if name else session.get("username", "User")

def require_login():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return None

def require_approved():
    resp = require_login()
    if resp:
        return resp
    if not session.get("approved", True) and session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    return None

def require_admin():
    resp = require_login()
    if resp:
        return resp
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    return None

def sanitize_course_slug(name):
    name = (name or "").strip()
    if not name:
        return "untitled"
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return (name[:60] or "untitled")

def list_report_folders(username):

    result = []
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT slug,
                   MAX(course)     AS course,
                   COUNT(*)        AS count,
                   MAX(created_at) AS latest_ts
            FROM reports
            WHERE username = %s
            GROUP BY slug
            ORDER BY MAX(created_at) DESC
        """, (username,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for row in rows:
            latest = row["latest_ts"]
            result.append({
                "slug":        row["slug"],
                "display":     row["course"] or row["slug"].replace("_", " "),
                "count":       row["count"],
                "latest_ts":   latest.isoformat() if latest else "",
            })
    except Exception as exc:
        print(f"[FASDA] list_report_folders error: {exc}")
    return result

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            session["logged_in"] = True
            session["username"] = username
            session["role"] = user["role"]
            session["user_id"] = user["id"]
            session["first_name"] = user.get("first_name") or username
            session["last_name"] = user.get("last_name") or ""

            approved_val = user.get("approved")
            session["approved"] = True if approved_val is None else bool(approved_val)
            if user["role"] == "admin":
                session["approved"] = True

            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        phone = request.form.get("phone", "").strip()

        if not username or not password:
            return render_template("register.html", error="Username and password are required.")

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return render_template("register.html", error="Username already exists.")

        try:
            cur.execute("""
                INSERT INTO users (username, password, first_name, last_name, phone, role, approved)
                VALUES (%s, %s, %s, %s, %s, 'user', FALSE)
            """, (username, password, first_name, last_name, phone))
        except Exception:
            conn.rollback()
            cur.execute("""
                INSERT INTO users (username, password, first_name, last_name, phone, role)
                VALUES (%s, %s, %s, %s, %s, 'user')
            """, (username, password, first_name, last_name, phone))

        conn.commit()
        cur.close()
        conn.close()

        return render_template(
            "register.html",
            success="Account created successfully! Please wait for admin approval before signing in."
        )

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    resp = require_login()
    if resp:
        return resp

    role = session.get("role", "user")
    approved = session.get("approved", True)
    display_name = get_display_name()

    if role == "admin":
        reports_summary = []
    else:
        reports_summary = list_report_folders(session.get("username", ""))

    return render_template(
        "dashboard.html",
        role=role,
        display_name=display_name,
        approved=approved,
        pending_approval=(not approved and role != "admin"),
        reports_summary=reports_summary,
    )

@app.route("/profile")
def profile():
    resp = require_login()
    if resp:
        return resp

    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE id = %s", (session.get("user_id"),))
    user = cur.fetchone()
    cur.close()
    conn.close()

    return render_template(
        "profile.html",
        user=user,
        display_name=get_display_name(),
        role=session.get("role", "user"),
        approved=session.get("approved", True),
    )

@app.route("/admin")
def admin_panel():
    resp = require_admin()
    if resp:
        return resp

    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    users = []
    try:
        cur.execute("""
            SELECT id, username, first_name, last_name, phone, role,
                   COALESCE(approved, TRUE) AS approved, created_at
            FROM users ORDER BY created_at DESC
        """)
        users = cur.fetchall()
    except Exception:
        conn.rollback()

        try:
            cur.execute("""
                SELECT id, username, first_name, last_name, phone, role, created_at
                FROM users ORDER BY created_at DESC
            """)
            users = cur.fetchall()
        except Exception:
            pass
    finally:
        cur.close()
        conn.close()

    return render_template(
        "admin.html",
        users=users,
        alpha=_app_config.ALPHA,
        display_name=get_display_name(),
        role="admin",
    )

@app.route("/admin/approve/<int:user_id>", methods=["POST"])
def admin_approve_user(user_id):
    resp = require_admin()
    if resp:
        return resp

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET approved = TRUE WHERE id = %s", (user_id,))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return redirect(url_for("admin_panel"))

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
def admin_delete_user(user_id):
    resp = require_admin()
    if resp:
        return resp

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM users WHERE id = %s AND role != 'admin'", (user_id,))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return redirect(url_for("admin_panel"))

@app.route("/admin/update_alpha", methods=["POST"])
def admin_update_alpha():
    resp = require_admin()
    if resp:
        return resp

    try:
        new_alpha = float(request.form.get("alpha", 0.4))
        new_alpha = max(0.0, min(1.0, round(new_alpha, 4)))

        _app_config.ALPHA = new_alpha

        config_path = os.path.join(BASE_DIR, "config.py")
        with open(config_path, "r") as f:
            content = f.read()
        content = re.sub(r"ALPHA\s*=\s*[\d.]+", f"ALPHA = {new_alpha}", content)
        with open(config_path, "w") as f:
            f.write(content)
    except Exception:
        pass

    return redirect(url_for("admin_panel"))

@app.route("/verify", methods=["GET", "POST"])
def verify():
    resp = require_approved()
    if resp:
        return resp

    result = None
    error = None

    if request.method == "POST":
        try:
            student_id = request.form.get("student_id", "").zfill(3)
            file = request.files.get("signature")

            if not file or not file.filename:
                error = "Please upload a signature image."
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                user_dir = os.path.join(DATASET_PATH, student_id)

                _t0 = time.perf_counter()
                cnn, hog, hybrid, decision = verify_signature(
                    model, student_id, user_dir, file_path
                )
                _verify_time = time.perf_counter() - _t0

                print(
                    f"\n[FASDA] ── Verify Timing ─────────────────────────────\n"
                    f"[FASDA]   Student ID     : {student_id}\n"
                    f"[FASDA]   Inference Time : {_verify_time:.3f} s\n"
                    f"[FASDA]   CNN score      : {cnn:.4f}\n"
                    f"[FASDA]   HOG score      : {hog:.4f}\n"
                    f"[FASDA]   Hybrid score   : {hybrid:.4f}\n"
                    f"[FASDA]   Decision       : {decision}\n"
                    f"[FASDA] ─────────────────────────────────────────────────\n",
                    flush=True,
                )

                result = {
                    "cnn": round(cnn, 3),
                    "hog": round(hog, 3),
                    "hybrid": round(hybrid, 3),
                    "decision": decision,
                }
        except Exception as e:
            error = str(e)

    return render_template(
        "verify.html",
        result=result,
        error=error,
        role=session.get("role", "user"),
        display_name=get_display_name(),
    )

@app.route("/attendance/flag", methods=["POST"])
def attendance_flag():
    if not session.get("logged_in"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        data = request.get_json(silent=True) or {}
        result = toggle_flag(
            signature_key=data.get("signature_key", ""),
            student_id=data.get("student_id", ""),
            student_name=data.get("student_name", ""),
            source_pdf=data.get("source_pdf", ""),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/attendance", methods=["GET", "POST"])
def attendance_upload():
    resp = require_approved()
    if resp:
        return resp

    if request.method == "GET":
        return render_template(
            "attendance_upload.html",
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

    file = request.files.get("pdf")
    if file is None or file.filename == "":
        return render_template(
            "attendance_upload.html",
            error="Please select a PDF file.",
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

    try:
        filename = secure_filename(file.filename)
        temp_pdf_path = os.path.join(ATTENDANCE_UPLOAD_FOLDER, filename)
        file.save(temp_pdf_path)

        saved_pdf_name = save_uploaded_pdf(temp_pdf_path)

        _route_start = time.perf_counter()

        _t0 = time.perf_counter()
        seg_result = process_attendance_pdf(temp_pdf_path)
        _seg_time = time.perf_counter() - _t0
        print(f"[FASDA] PDF Segmentation     : {_seg_time:.3f}s")

        _t0 = time.perf_counter()
        analysis_result = analyze_and_annotate(
            seg_result["metadata_csv"],
            seg_result["output_dir"],
            RENDERED_PAGES_FOLDER,
        )
        _ml_time = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        raw_pages       = analysis_result.get("pages", [])
        results         = analysis_result.get("results", [])
        normalized_pages = normalize_page_paths(raw_pages)
        grouped_results  = group_results_by_student(results)
        results_json     = safe_json_results(results)
        _post_time = time.perf_counter() - _t0

        _total_time = time.perf_counter() - _route_start
        print(
            f"\n[FASDA] ── Route Timing Summary ─────────────────────────────\n"
            f"[FASDA]   PDF Segmentation   : {_seg_time:>8.3f} s\n"
            f"[FASDA]   ML Analysis        : {_ml_time:>8.3f} s\n"
            f"[FASDA]   Result Processing  : {_post_time:>8.3f} s\n"
            f"[FASDA]   ─────────────────────────────────────────────────\n"
            f"[FASDA]   TOTAL              : {_total_time:>8.3f} s\n"
            f"[FASDA] ─────────────────────────────────────────────────────\n",
            flush=True,
        )

        return render_template(
            "attendance_result.html",
            pages=normalized_pages,
            results=results,
            grouped_results=grouped_results,
            results_json=results_json,
            flagged_map=get_flagged_map(),
            source_pdf=saved_pdf_name,
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

    except Exception as e:
        print("ATTENDANCE ERROR:", str(e))
        return render_template(
            "attendance_upload.html",
            error=f"Attendance processing failed: {str(e)}",
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

@app.route("/reports/folders")
def report_folders_api():
    if not session.get("logged_in"):
        return jsonify([]), 401
    folders = list_report_folders(session.get("username", ""))
    return jsonify([{"slug": f["slug"], "display": f["display"]} for f in folders])

@app.route("/reports/save", methods=["POST"])
def save_report():
    if not session.get("logged_in"):
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    course_input = data.get("course", "").strip()
    slug        = sanitize_course_slug(course_input)
    username    = session.get("username", "unknown")

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO reports (username, course, slug, source_pdf, summary, results)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
            RETURNING id
        """, (
            username,
            course_input or "Untitled",
            slug,
            data.get("source_pdf", ""),
            json.dumps(data.get("summary", {})),
            json.dumps(data.get("results", [])),
        ))
        report_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({
            "success":       True,
            "slug":          slug,
            "report_id":     report_id,
            "display_course": course_input or "Untitled",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/reports/<course_slug>")
def list_course_reports(course_slug):
    resp = require_login()
    if resp:
        return resp

    course_slug = re.sub(r"[^\w_-]", "", course_slug)
    username    = session.get("username", "")

    try:
        conn = get_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT id, course, slug, source_pdf, summary, created_at, username AS created_by
            FROM   reports
            WHERE  username = %s AND slug = %s
            ORDER  BY created_at DESC
        """, (username, course_slug))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[FASDA] list_course_reports error: {exc}")
        rows = []

    if not rows:
        return redirect(url_for("dashboard"))

    reports = []
    agg = {"total": 0, "genuine": 0, "forged": 0, "unknown": 0}
    course_display = course_slug.replace("_", " ")

    for row in rows:
        summary = row["summary"] or {}
        if isinstance(summary, str):
            try: summary = json.loads(summary)
            except Exception: summary = {}
        if not reports:
            course_display = row["course"] or course_display
        reports.append({
            "id":         row["id"],
            "course":     row["course"] or course_display,
            "created_at": row["created_at"].isoformat() if row["created_at"] else "",
            "created_by": row["created_by"] or "",
            "source_pdf": row["source_pdf"] or "",
            "summary":    summary,
            "slug":       course_slug,
        })
        agg["total"]   += int(summary.get("total",   0) or 0)
        agg["genuine"] += int(summary.get("genuine", 0) or 0)
        agg["forged"]  += int(summary.get("forged",  0) or 0)
        agg["unknown"] += int(summary.get("unknown", 0) or 0)

    return render_template(
        "report_list.html",
        course_slug=course_slug,
        course_display=course_display,
        reports=reports,
        aggregate=agg,
        display_name=get_display_name(),
        role=session.get("role", "user"),
    )

@app.route("/reports/<course_slug>/<int:report_id>")
def view_report(course_slug, report_id):
    resp = require_login()
    if resp:
        return resp

    course_slug = re.sub(r"[^\w_-]", "", course_slug)
    username    = session.get("username", "")

    try:
        conn = get_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT id, course, slug, source_pdf, summary, results, created_at, username AS created_by
            FROM   reports
            WHERE  id = %s AND username = %s
        """, (report_id, username))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        return "Failed to load report", 500

    if not row:
        return "Report not found", 404

    summary = row["summary"] or {}
    if isinstance(summary, str):
        try: summary = json.loads(summary)
        except Exception: summary = {}

    results = row["results"] or []
    if isinstance(results, str):
        try: results = json.loads(results)
        except Exception: results = []

    report = {
        "id":         row["id"],
        "course":     row["course"],
        "slug":       row["slug"],
        "source_pdf": row["source_pdf"],
        "summary":    summary,
        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
        "created_by": row["created_by"] or "",
    }

    return render_template(
        "report_view.html",
        report=report,
        results_json=json.dumps(results),
        display_name=get_display_name(),
        role=session.get("role", "user"),
    )

@app.route("/plots")
def plots():
    resp = require_approved()
    if resp:
        return resp

    try:
        plot_similarity_distributions()
        return render_template(
            "plots.html",
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )
    except Exception as e:
        return render_template(
            "plots.html",
            error=str(e),
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

@app.route("/benchmark")
def benchmark():
    resp = require_admin()
    if resp:
        return resp

    try:
        evaluate_distributions()
        return render_template(
            "plots.html",
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )
    except Exception as e:
        return render_template(
            "plots.html",
            error=str(e),
            role=session.get("role", "user"),
            display_name=get_display_name(),
        )

if __name__ == "__main__":
    app.run(debug=True)
