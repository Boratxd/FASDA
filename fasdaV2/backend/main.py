# main.py
import os
import tensorflow as tf

from config import DATASET_PATH, MODEL_PATH
from ml.database import init_db
from ml.cnn_model import L1DistanceLayer, build_siamese_model
from ml.dataset_utils import extract_user_signatures
from ml.verification import verify_signature
from ml.evaluation import plot_similarity_distributions, evaluate_distributions

import random
import numpy as np
from ml.cnn_model import load_image

# =========================================================
# TRAINING (same logic as your current version)
# =========================================================
def train_model():
    pairs, labels = [], []
    users = [f"{i:03d}" for i in range(1,401)]
    random.shuffle(users)

    for uid in users[:200]:
        udir = os.path.join(DATASET_PATH, uid)
        if not os.path.isdir(udir):
            continue

        genuine, forged = extract_user_signatures(udir)
        if len(genuine) < 2 or not forged:
            continue

        g_imgs = [load_image(os.path.join(udir,g)) for g in genuine]

        for i in range(len(g_imgs)):
            for j in range(i+1, len(g_imgs)):
                pairs.append([g_imgs[i], g_imgs[j]])
                labels.append(0)

        for f in forged:
            pairs.append([g_imgs[0], load_image(os.path.join(udir,f))])
            labels.append(1)

    pairs = np.array(pairs)
    labels = np.array(labels)

    model = build_siamese_model()
    model.fit([pairs[:,0], pairs[:,1]], labels,
              epochs=10, batch_size=32, validation_split=0.2)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print("[INFO] Model trained & saved:", MODEL_PATH)

# =========================================================
# MAIN
# =========================================================
def main():
    init_db()

    print("=== FASDA HYBRID SIGNATURE VERIFICATION ===")
    print("1 - Train Siamese CNN")
    print("2 - Verify Single Signature")
    print("3 - Plot Similarity Distributions (from SQLite)")
    print("4 - Generate Distribution Graphs (Benchmark)")
    choice = input("Select: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model not found: {MODEL_PATH}")
            print("Train first (option 1) or place gpds_siamese_modelv4.h5 into models/ folder.")
            return

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"L1DistanceLayer": L1DistanceLayer}
        )

        sid = input("Student ID (001–400): ").zfill(3)
        test = input("Signature image path: ")

        udir = os.path.join(DATASET_PATH, sid)
        cnn, hog, final, decision = verify_signature(model, sid, udir, test)

        print("\n===== HYBRID VERIFICATION RESULT =====")
        print("CNN similarity :", round(cnn, 3))
        print("HOG similarity :", round(hog, 3))
        print("Final score    :", round(final, 3))
        print("Decision       :", decision)

    elif choice == "3":
        plot_similarity_distributions()

    elif choice == "4":
        evaluate_distributions()

    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
