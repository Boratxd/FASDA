import os
import psycopg2
import psycopg2.extras
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.utils import secure_filename

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "gpds_siamese_modelv4.h5")
DATASET_PATH = r"C:\Users\alper\Desktop\CNG 491\DataSet\SignatureGPDSSyntheticSignaturesManuscriptsv\firmasSINTESISmanuscritas"

os.makedirs(VERIFY_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RENDERED_PAGES_FOLDER, exist_ok=True)

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

        cur.execute("""
            INSERT INTO users (username, password, first_name, last_name, phone, role)
            VALUES (%s, %s, %s, %s, %s, 'user')
        """, (username, password, first_name, last_name, phone))
        conn.commit()
        cur.close()
        conn.close()

        return render_template("register.html", success="Account created successfully! You can now sign in.")

    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("dashboard.html", role=session.get("role", "user"))


@app.route("/verify", methods=["GET", "POST"])
def verify():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

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

                cnn, hog, hybrid, decision = verify_signature(
                    model, student_id, user_dir, file_path
                )

                result = {
                    "cnn": round(cnn, 3),
                    "hog": round(hog, 3),
                    "hybrid": round(hybrid, 3),
                    "decision": decision
                }
        except Exception as e:
            error = str(e)

    return render_template(
        "verify.html",
        result=result,
        error=error,
        role=session.get("role", "user")
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
            source_pdf=data.get("source_pdf", "")
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/attendance", methods=["GET", "POST"])
def attendance_upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("attendance_upload.html", role=session.get("role", "user"))

    file = request.files.get("pdf")
    if file is None or file.filename == "":
        return render_template(
            "attendance_upload.html",
            error="Please select a PDF file.",
            role=session.get("role", "user")
        )

    try:
        filename = secure_filename(file.filename)
        temp_pdf_path = os.path.join(ATTENDANCE_UPLOAD_FOLDER, filename)
        file.save(temp_pdf_path)

        saved_pdf_name = save_uploaded_pdf(temp_pdf_path)

        seg_result = process_attendance_pdf(temp_pdf_path)
        analysis_result = analyze_and_annotate(
            seg_result["metadata_csv"],
            seg_result["output_dir"],
            RENDERED_PAGES_FOLDER
        )

        raw_pages = analysis_result.get("pages", [])
        results = analysis_result.get("results", [])
        normalized_pages = normalize_page_paths(raw_pages)

        return render_template(
            "attendance_result.html",
            pages=normalized_pages,
            results=results,
            flagged_map=get_flagged_map(),
            source_pdf=saved_pdf_name,
            role=session.get("role", "user")
        )

    except Exception as e:
        print("ATTENDANCE ERROR:", str(e))
        return render_template(
            "attendance_upload.html",
            error=f"Attendance processing failed: {str(e)}",
            role=session.get("role", "user")
        )


@app.route("/plots")
def plots():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        plot_similarity_distributions()
        return render_template("plots.html", role=session.get("role", "user"))
    except Exception as e:
        return render_template("plots.html", error=str(e), role=session.get("role", "user"))


@app.route("/benchmark")
def benchmark():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))

    try:
        evaluate_distributions()
        return render_template("plots.html", role=session.get("role", "user"))
    except Exception as e:
        return render_template("plots.html", error=str(e), role=session.get("role", "user"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
