# ml/hog_features.py
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
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def segment_signature(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 200]
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return binary[y:y+h, x:x+w]

def extract_features(sig):
    img = cv2.resize(sig, IMG_SIZE).astype("float32") / 255.0
    hog_feat = hog(img, 9, (8,8), (2,2), block_norm="L2-Hys")
    density = np.sum(img > 0) / img.size
    return np.hstack([hog_feat, density])

def classical_similarity(ref, test):
    r = extract_features(segment_signature(preprocess_image(ref)))
    t = extract_features(segment_signature(preprocess_image(test)))
    return 1 - cosine(r, t)
