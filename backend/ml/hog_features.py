import cv2
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import cosine
from config import IMG_SIZE

# =========================================================
# CLASSICAL HOG
# =========================================================
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image for HOG: {path}")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def segment_signature(binary):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # keep only meaningful contours
    contours = [c for c in contours if cv2.contourArea(c) >= 200]

    # SAFE FALLBACK: if nothing found, return full binary image
    if not contours:
        return binary

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return binary[y:y+h, x:x+w]


def extract_features(sig):
    img = cv2.resize(sig, IMG_SIZE).astype("float32") / 255.0

    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    density = np.sum(img > 0) / img.size
    return np.hstack([hog_feat, density])


def classical_similarity(ref, test):
    try:
        r = extract_features(segment_signature(preprocess_image(ref)))
        t = extract_features(segment_signature(preprocess_image(test)))

        sim = 1 - cosine(r, t)

        # guard against nan
        if np.isnan(sim):
            return 0.0

        return float(sim)

    except Exception as e:
        print(f"[WARN] HOG failed for comparison:\n  ref={ref}\n  test={test}\n  error={e}")
        return 0.0
