from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from additional.flag_store import create_flag, get_flagged_map, save_uploaded_pdf

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        pdf_file = request.files.get("pdf")
        if pdf_file is None or pdf_file.filename == "":
            return "No PDF file selected", 400

        if not allowed_file(pdf_file.filename):
            return "Only PDF files are allowed", 400

        filename = secure_filename(pdf_file.filename)
        stored_path = save_uploaded_pdf(pdf_file, filename=filename)

        return f"Saved to {stored_path}", 200

    return render_template("attendance_upload.html")

@app.route("/attendance/flag", methods=["POST"])
def flag_attendance_row():
    payload = request.get_json(silent=True) or {}

    required_fields = ["page", "row", "student_id"]
    for field in required_fields:
        if str(payload.get(field, "")).strip() == "":
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        result = create_flag(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify(result), 200

@app.route("/attendance/result-demo")
def attendance_result_demo():
    pages = {1: "page_1.jpg", 2: "page_2.jpg"}
    results = [
        {
            "page": 1,
            "row": 3,
            "student_id": "2385664",
            "cnn": "0.82",
            "hog": "0.79",
            "hybrid": "0.81",
            "label": "FORGED",
        },
        {
            "page": 1,
            "row": 4,
            "student_id": "2385012",
            "cnn": "0.95",
            "hog": "0.93",
            "hybrid": "0.94",
            "label": "GENUINE",
        },
    ]
    return render_template(
        "attendance_result.html",
        pages=pages,
        results=results,
        flagged_map=get_flagged_map(),
    )

if __name__ == "__main__":
    app.run(debug=True)
