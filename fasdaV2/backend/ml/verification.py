# ml/verification.py
import os
import numpy as np

from config import ALPHA, THRESHOLD
from ml.dataset_utils import extract_user_signatures
from ml.database import log_result
from ml.cnn_model import load_image
from ml.hog_features import classical_similarity

# =========================================================
# VERIFICATION (WITH SQLITE LOGGING)
# =========================================================
def verify_signature(model, student_id, user_dir, test_path):
    genuine, _ = extract_user_signatures(user_dir)
    genuine = genuine[:5]

    cnn_scores, hog_scores = [], []
    test_img = load_image(test_path)

    for g in genuine:
        ref_path = os.path.join(user_dir, g)
        ref_img = load_image(ref_path)

        prob_fake = model.predict(
            [ref_img[None], test_img[None]], verbose=0
        )[0][0]

        cnn_scores.append(1 - prob_fake)
        hog_scores.append(classical_similarity(ref_path, test_path))

    cnn_m = np.mean(cnn_scores)
    hog_m = np.mean(hog_scores)
    hybrid = ALPHA * cnn_m + (1 - ALPHA) * hog_m
    label = "GENUINE" if hybrid >= THRESHOLD else "FORGED"

    log_result(student_id, test_path, label, cnn_m, hog_m, hybrid)

    return cnn_m, hog_m, hybrid, label
