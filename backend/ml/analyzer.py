import os
import time
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

from config import MODEL_PATH, ALPHA, THRESHOLD
from ml.cnn_model import L1DistanceLayer, load_image
from ml.hog_features import classical_similarity, get_feature_vector
from ml.perf import PerfTimer

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"L1DistanceLayer": L1DistanceLayer}
)

_ANALYSIS_ALPHA = 0.15

def compare_two_signatures(ref_path, test_path, alpha=None):
    if alpha is None:
        alpha = _ANALYSIS_ALPHA

    ref_img  = load_image(ref_path)
    test_img = load_image(test_path)

    prob_fake = model.predict(
        [ref_img[None], test_img[None]], verbose=0
    )[0][0]

    cnn_sim = float(1.0 - prob_fake)
    hog_sim = float(classical_similarity(ref_path, test_path))
    hybrid  = alpha * cnn_sim + (1.0 - alpha) * hog_sim

    return cnn_sim, hog_sim, float(hybrid)

def median_decision(feature_vecs, fallback_scores):

    n = len(feature_vecs)
    if n < 2:
        return ["GENUINE"] * n, list(fallback_scores)

    Z_THRESHOLD = -0.8
    MAX_FORGED  = 0.60

    valid = [(i, v) for i, v in enumerate(feature_vecs) if v is not None]

    if len(valid) < 2:

        return ["GENUINE"] * n, list(fallback_scores)

    mat    = np.vstack([v for _, v in valid])
    median = np.median(mat, axis=0)

    scores = [None] * n
    for i, feat in valid:
        sim = 1.0 - cosine(feat, median)
        if np.isnan(sim):
            sim = 0.0
        scores[i] = float(np.clip(sim, 0.0, 1.0))

    valid_scores_list = [s for s in scores if s is not None]
    penalty = float(np.min(valid_scores_list)) if valid_scores_list else 0.0
    scores  = [s if s is not None else penalty for s in scores]

    arr  = np.array(scores, dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    decisions = []
    for s in scores:
        z = (s - mean) / std if std > 1e-6 else 0.0
        decisions.append("FORGED" if z < Z_THRESHOLD else "GENUINE")

    if decisions.count("FORGED") / n > MAX_FORGED:
        decisions = [
            "GENUINE" if s >= THRESHOLD else "FORGED"
            for s in fallback_scores
        ]

    return decisions, scores

def analyze_and_annotate(metadata_csv, cropped_dir, rendered_pages_dir):
    perf = PerfTimer("ML Analysis Pipeline")

    df      = pd.read_csv(metadata_csv)
    grouped = df.groupby(["page", "row", "student_id"])

    annotated_pages = {}
    results         = []

    t_hog      = 0.0
    t_cnn      = 0.0
    t_decision = 0.0
    t_annotate = 0.0

    total_sigs   = 0
    total_pairs  = 0
    total_rows   = 0
    n_genuine    = 0
    n_forged     = 0

    for (page_num, row_num, student_id), group in grouped:
        group = group.sort_values("cell_index")

        signature_paths = []
        for _, row in group.iterrows():
            sig_path = os.path.join(cropped_dir, row["file_name"])
            if os.path.exists(sig_path):
                signature_paths.append((row, sig_path))

        if len(signature_paths) < 2:
            continue

        page_path = os.path.join(rendered_pages_dir, f"page_{int(page_num)}.jpg")
        page_img  = cv2.imread(page_path)
        if page_img is None:
            continue

        n = len(signature_paths)
        total_rows += 1
        total_sigs += n

        _t = time.perf_counter()
        feat_vecs = []
        for _, sig_path in signature_paths:
            feat_vecs.append(get_feature_vector(sig_path))
        t_hog += time.perf_counter() - _t

        _t = time.perf_counter()
        cnn_matrix    = np.zeros((n, n))
        hog_matrix    = np.zeros((n, n))
        hybrid_matrix = np.zeros((n, n))

        for idx in range(n):
            for jdx in range(n):
                if idx == jdx:
                    continue
                _, ref_path  = signature_paths[jdx]
                _, test_path = signature_paths[idx]
                try:
                    cnn, hog, hyb = compare_two_signatures(ref_path, test_path)
                    cnn_matrix[idx, jdx]    = cnn
                    hog_matrix[idx, jdx]    = hog
                    hybrid_matrix[idx, jdx] = hyb
                    total_pairs += 1
                except Exception as e:
                    print(f"[WARN] Compare failed idx={idx} jdx={jdx}: {e}")
        t_cnn += time.perf_counter() - _t

        cnn_means    = []
        hog_means    = []
        hybrid_means = []
        for idx in range(n):
            mask = [j for j in range(n) if j != idx]
            cnn_means.append(float(np.mean(cnn_matrix[idx, mask])))
            hog_means.append(float(np.mean(hog_matrix[idx, mask])))
            hybrid_means.append(float(np.mean(hybrid_matrix[idx, mask])))

        _t = time.perf_counter()
        decisions, median_scores = median_decision(feat_vecs, hybrid_means)
        t_decision += time.perf_counter() - _t

        n_genuine += decisions.count("GENUINE")
        n_forged  += decisions.count("FORGED")

        _t = time.perf_counter()
        page_modified = False

        for idx, (meta_row, _) in enumerate(signature_paths):
            decision    = decisions[idx]
            cnn_mean    = round(cnn_means[idx],      3)
            hog_mean    = round(hog_means[idx],      3)
            hybrid_mean = round(hybrid_means[idx],   3)
            med_score   = round(median_scores[idx],  3)

            x = int(meta_row["x"])
            y = int(meta_row["y"])

            crop_img = cv2.imread(signature_paths[idx][1])
            h, w = (crop_img.shape[:2] if crop_img is not None else (60, 120))

            color = (0, 0, 220) if decision == "FORGED" else (0, 200, 60)
            cv2.rectangle(page_img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                page_img,
                f"{decision} {med_score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            page_modified = True

            results.append({
                "student_id":   str(student_id),
                "page":         int(page_num),
                "row":          int(row_num),
                "page_number":  int(page_num),
                "row_index":    int(row_num),

                "cnn":          cnn_mean,
                "hog":          hog_mean,
                "hybrid":       hybrid_mean,

                "cnn_score":    cnn_mean,
                "hog_score":    hog_mean,
                "hybrid_score": hybrid_mean,
                "similarity":   hybrid_mean,

                "decision":     decision,
                "label":        decision,

                "signature_key": str(meta_row["file_name"]),
                "file_name":     meta_row["file_name"],
            })

        if page_modified:
            cv2.imwrite(page_path, page_img)
            annotated_pages[int(page_num)] = page_path

        t_annotate += time.perf_counter() - _t

    perf.stages = [
        ("HOG Feature Extraction",
         t_hog,
         f"{total_sigs} signatures"),
        ("CNN Pairwise Comparisons",
         t_cnn,
         f"{total_pairs} pairs"),
        ("Median Decision",
         t_decision,
         f"{total_rows} rows"),
        ("Page Annotation",
         t_annotate,
         f"{len(annotated_pages)} pages"),
    ]

    forgery_rate = (
        f"{n_forged / (n_genuine + n_forged) * 100:.1f} %"
        if (n_genuine + n_forged) > 0 else "N/A"
    )

    perf.print_report(
        signatures_processed = total_sigs,
        rows_analyzed        = total_rows,
        cnn_hog_pairs        = total_pairs,
        genuine              = n_genuine,
        forged               = n_forged,
        forgery_rate         = forgery_rate,
    )

    return {
        "pages":   annotated_pages,
        "results": results,
    }
