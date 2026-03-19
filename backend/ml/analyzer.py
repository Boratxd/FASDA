import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, ALPHA, THRESHOLD
from ml.cnn_model import L1DistanceLayer, load_image
from ml.hog_features import classical_similarity

# Load model once
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"L1DistanceLayer": L1DistanceLayer}
)


def compare_two_signatures(ref_path, test_path):
    ref_img = load_image(ref_path)
    test_img = load_image(test_path)

    prob_fake = model.predict(
        [ref_img[None], test_img[None]],
        verbose=0
    )[0][0]

    cnn_similarity = 1 - prob_fake
    hog_similarity = classical_similarity(ref_path, test_path)
    hybrid_score = ALPHA * cnn_similarity + (1 - ALPHA) * hog_similarity

    return float(cnn_similarity), float(hog_similarity), float(hybrid_score)


def analyze_and_annotate(metadata_csv, cropped_dir, rendered_pages_dir):
    df = pd.read_csv(metadata_csv)

    grouped = df.groupby(["page", "row", "student_id"])

    annotated_pages = {}
    results = []

    for (page_num, row_num, student_id), group in grouped:
        group = group.sort_values("cell_index")

        signature_paths = []
        for _, row in group.iterrows():
            sig_path = os.path.join(cropped_dir, row["file_name"])
            if os.path.exists(sig_path):
                signature_paths.append((row, sig_path))

        # Need at least 2 signatures in same row to compare
        if len(signature_paths) < 2:
            continue

        page_path = os.path.join(rendered_pages_dir, f"page_{int(page_num)}.jpg")
        page_img = cv2.imread(page_path)
        if page_img is None:
            continue

        page_modified = False

        for idx, (meta_row, test_path) in enumerate(signature_paths):
            cnn_scores = []
            hog_scores = []
            hybrid_scores = []

            for jdx, (_, ref_path) in enumerate(signature_paths):
                if idx == jdx:
                    continue

                try:
                    cnn, hog, hybrid = compare_two_signatures(ref_path, test_path)
                    cnn_scores.append(cnn)
                    hog_scores.append(hog)
                    hybrid_scores.append(hybrid)
                except Exception as e:
                    print(f"[WARN] Comparison failed: ref={ref_path}, test={test_path}, error={e}")

            if not hybrid_scores:
                continue

            cnn_mean = float(np.mean(cnn_scores))
            hog_mean = float(np.mean(hog_scores))
            hybrid_mean = float(np.mean(hybrid_scores))

            decision = "FORGED" if hybrid_mean < THRESHOLD else "GENUINE"

            x = int(meta_row["x"])
            y = int(meta_row["y"])

            crop_img = cv2.imread(test_path)
            if crop_img is not None:
                h, w = crop_img.shape[:2]
            else:
                w, h = 120, 60

            color = (0, 0, 255) if decision == "FORGED" else (0, 255, 0)

            cv2.rectangle(page_img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                page_img,
                f"{decision} {hybrid_mean:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            page_modified = True

            # UI-compatible keys
            results.append({
                "student_id": str(student_id),
                "page": int(page_num),
                "row": int(row_num),
                "page_number": int(page_num),
                "row_index": int(row_num),

                "cnn": round(cnn_mean, 3),
                "hog": round(hog_mean, 3),
                "hybrid": round(hybrid_mean, 3),

                # old compatibility fields
                "cnn_score": round(cnn_mean, 3),
                "hog_score": round(hog_mean, 3),
                "hybrid_score": round(hybrid_mean, 3),
                "similarity": round(hybrid_mean, 3),

                "decision": decision,
                "label": decision,

                # stable unique key for flagging
                "signature_key": str(meta_row["file_name"]),
                "file_name": meta_row["file_name"]
            })

        if page_modified:
            cv2.imwrite(page_path, page_img)
            annotated_pages[int(page_num)] = page_path

    return {
        "pages": annotated_pages,
        "results": results
    }
