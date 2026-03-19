import os
import random
import psycopg2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_PATH, DATASET_PATH, PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
from ml.cnn_model import L1DistanceLayer
from ml.dataset_utils import extract_user_signatures
from ml.verification import verify_signature

sns.set(style="whitegrid")

# =========================================================
# PLOT DIRECTORY (WEB SAFE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "..", "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# HELPER: SAVE KDE PLOT
# =========================================================
def save_plot(genuine, impostor, title, filename):
    path = os.path.join(PLOT_DIR, filename)

    if os.path.exists(path):
        print(f"[INFO] {filename} already exists. Skipping.")
        return

    plt.figure(figsize=(8, 5))
    sns.kdeplot(genuine, label="Genuine", fill=True)
    sns.kdeplot(impostor, label="Impostor", fill=True)
    plt.title(title)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


# =========================================================
# SQLITE-BASED DISTRIBUTIONS (FROM LOGS)
# =========================================================
def plot_similarity_distributions():
    cnn_path = os.path.join(PLOT_DIR, "cnn_distribution.png")
    hog_path = os.path.join(PLOT_DIR, "hog_distribution.png")
    hyb_path = os.path.join(PLOT_DIR, "hybrid_distribution.png")

    # 🔒 If already generated → DO NOTHING
    if os.path.exists(cnn_path) and os.path.exists(hog_path) and os.path.exists(hyb_path):
        return

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DATABASE,
        user=PG_USER, password=PG_PASSWORD
    )
    c = conn.cursor()

    c.execute("""
        SELECT cnn_similarity, hog_similarity, hybrid_score, label
        FROM verification_logs
    """)
    rows = c.fetchall()
    c.close()
    conn.close()

    cnn_g, cnn_i = [], []
    hog_g, hog_i = [], []
    hyb_g, hyb_i = [], []

    for cnn, hog, hyb, label in rows:
        if label == "GENUINE":
            cnn_g.append(cnn)
            hog_g.append(hog)
            hyb_g.append(hyb)
        else:
            cnn_i.append(cnn)
            hog_i.append(hog)
            hyb_i.append(hyb)

    if not cnn_g or not cnn_i:
        print("[WARN] Not enough data in SQLite for plotting.")
        return

    save_plot(
        cnn_g, cnn_i,
        "CNN Similarity Distribution (Genuine vs Impostor)",
        "cnn_distribution.png"
    )

    save_plot(
        hog_g, hog_i,
        "HOG Similarity Distribution (Genuine vs Impostor)",
        "hog_distribution.png"
    )

    save_plot(
        hyb_g, hyb_i,
        "Hybrid Similarity Distribution (Genuine vs Impostor)",
        "hybrid_distribution.png"
    )

    print("[INFO] SQLite-based plots saved.")

# =========================================================
# ACADEMIC BENCHMARK DISTRIBUTIONS
# =========================================================
def evaluate_distributions(num_users=50):
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"L1DistanceLayer": L1DistanceLayer}
    )

    users = [f"{i:03d}" for i in range(1, 401)]
    random.shuffle(users)
    users = users[:num_users]

    cnn_g, cnn_i = [], []
    hog_g, hog_i = [], []
    hyb_g, hyb_i = [], []

    for uid in users:
        udir = os.path.join(DATASET_PATH, uid)
        if not os.path.isdir(udir):
            continue

        genuine, _ = extract_user_signatures(udir)
        if len(genuine) < 2:
            continue

        # ---------- Genuine pair ----------
        test = os.path.join(udir, genuine[1])
        c, h, f, _ = verify_signature(model, uid, udir, test)
        cnn_g.append(c)
        hog_g.append(h)
        hyb_g.append(f)

        # ---------- Impostor pair ----------
        other = random.choice([u for u in users if u != uid])
        odir = os.path.join(DATASET_PATH, other)
        og, _ = extract_user_signatures(odir)
        if not og:
            continue

        test = os.path.join(odir, og[0])
        c, h, f, _ = verify_signature(model, uid, udir, test)
        cnn_i.append(c)
        hog_i.append(h)
        hyb_i.append(f)

    save_plot(
        cnn_g, cnn_i,
        "CNN Similarity Distribution (Benchmark)",
        "cnn_benchmark.png"
    )

    save_plot(
        hog_g, hog_i,
        "HOG Similarity Distribution (Benchmark)",
        "hog_benchmark.png"
    )

    save_plot(
        hyb_g, hyb_i,
        "Hybrid Similarity Distribution (Benchmark)",
        "hybrid_benchmark.png"
    )

    print("[INFO] Benchmark plots saved.")
