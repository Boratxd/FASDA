# ml/evaluation.py
import os
import random
import sqlite3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from config import DB_PATH, MODEL_PATH, DATASET_PATH
from ml.cnn_model import L1DistanceLayer
from ml.dataset_utils import extract_user_signatures
from ml.verification import verify_signature

sns.set(style="whitegrid")

def plot_similarity_distributions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT cnn_similarity, hog_similarity, hybrid_score, label
        FROM verification_logs
    """)
    rows = c.fetchall()
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

    def plot(genuine, impostor, title):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(genuine, label="Genuine", fill=True)
        sns.kdeplot(impostor, label="Impostor", fill=True)
        plt.title(title)
        plt.xlabel("Similarity Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    plot(cnn_g, cnn_i, "CNN Similarity Distribution (Genuine vs Impostor)")
    plot(hog_g, hog_i, "HOG Similarity Distribution (Genuine vs Impostor)")
    plot(hyb_g, hyb_i, "Hybrid Similarity Distribution (Genuine vs Impostor)")

# =========================================================
# DISTRIBUTION EVALUATION (ACADEMIC BENCHMARK)
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
        cnn_g.append(c); hog_g.append(h); hyb_g.append(f)

        # ---------- Impostor pair ----------
        other = random.choice([u for u in users if u != uid])
        odir = os.path.join(DATASET_PATH, other)
        og, _ = extract_user_signatures(odir)
        if not og:
            continue

        test = os.path.join(odir, og[0])
        c, h, f, _ = verify_signature(model, uid, udir, test)
        cnn_i.append(c); hog_i.append(h); hyb_i.append(f)

    def plot(g, i, title):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(g, label="Genuine", fill=True)
        sns.kdeplot(i, label="Impostor", fill=True)
        plt.title(title)
        plt.xlabel("Similarity Score")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    plot(cnn_g, cnn_i, "CNN Similarity Distribution (Genuine vs Impostor)")
    plot(hog_g, hog_i, "HOG Similarity Distribution (Genuine vs Impostor)")
    plot(hyb_g, hyb_i, "Hybrid Similarity Distribution (Genuine vs Impostor)")
