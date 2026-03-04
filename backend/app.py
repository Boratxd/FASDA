import os
from flask import Flask, render_template, request, redirect, session, url_for
import tensorflow as tf

from ml.verification import verify_signature
from ml.evaluation import plot_similarity_distributions, evaluate_distributions
from ml.custom_layers import L1DistanceLayer

# =============================
# CONFIG
# =============================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))





PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gpds_siamese_modelv4.h5")


DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "DataSet",
    "SignatureGPDSSyntheticSignaturesManuscriptsv",
    "firmasSINTESISmanuscritas"
)
DATASET_PATH = r"C:\Users\alper\Desktop\CNG 491\DataSet\SignatureGPDSSyntheticSignaturesManuscriptsv\firmasSINTESISmanuscritas"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================
# FLASK INIT
# =============================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)

app.secret_key = "fasda_demo_secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("=== FLASK PATH DEBUG ===")
print("BASE_DIR:", BASE_DIR)
print("STATIC_FOLDER:", app.static_folder)
print("TEMPLATE_FOLDER:", app.template_folder)

# =============================
# LOAD MODEL ONCE
# =============================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"L1DistanceLayer": L1DistanceLayer}
)


print(os.listdir("static/plots"))


# =============================
# LOGIN (DEMO)
# =============================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
    return render_template("login.html")

# =============================
# DASHBOARD
# =============================
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("dashboard.html")

# =============================
# VERIFY SIGNATURE
# =============================
@app.route("/verify", methods=["GET", "POST"])
def verify():
    if not session.get("logged_in"):
        return redirect("/")

    result = None

    if request.method == "POST":
        student_id = request.form["student_id"].zfill(3)
        file = request.files["signature"]

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
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

    return render_template("verify.html", result=result)

# =============================
# PLOTS
# =============================
@app.route("/plots")
def plots():
    if not session.get("logged_in"):
        return redirect("/")

    plot_similarity_distributions()
    return render_template("plots.html")

@app.route("/benchmark")
def benchmark():
    if not session.get("logged_in"):
        return redirect("/")

    evaluate_distributions()
    return render_template("plots.html")

# =============================
# LOGOUT
# =============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(debug=True)
