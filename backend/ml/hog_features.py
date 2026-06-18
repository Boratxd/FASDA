import cv2
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import cosine
from config import IMG_SIZE

def preprocess_image(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image for HOG: {path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    blur = cv2.GaussianBlur(img, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10

    )

    kernel_open = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    return binary

def segment_signature(binary):

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [c for c in contours if cv2.contourArea(c) >= 30]

    if not contours:
        return binary

    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    pad_x = max(3, int(w * 0.05))
    pad_y = max(3, int(h * 0.05))
    H, W = binary.shape
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return binary[y1:y2, x1:x2]

def resize_with_padding(img, target_size):

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype=np.float32)

    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    offset_y = (target_h - new_h) // 2
    offset_x = (target_w - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return canvas

def extract_features(sig):

    img = resize_with_padding(sig, IMG_SIZE).astype("float32") / 255.0

    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return feat

def get_feature_vector(path):

    try:
        binary = preprocess_image(path)
        seg    = segment_signature(binary)

        hog_feat = extract_features(seg)

        canvas   = resize_with_padding(seg, IMG_SIZE)
        ink_ratio = float(np.sum(canvas > 0) / canvas.size)

        canvas_u8 = canvas.astype(np.uint8)
        contours, _ = cv2.findContours(
            canvas_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        n_meaningful = len([c for c in contours if cv2.contourArea(c) >= 20])
        contour_norm  = float(min(n_meaningful / 30.0, 1.0))

        return np.concatenate([hog_feat, [ink_ratio * 3.0, contour_norm * 3.0]])
    except Exception as e:
        print(f"[WARN] get_feature_vector failed: {path}: {e}")
        return None

def classical_similarity(ref, test):

    try:
        r = extract_features(segment_signature(preprocess_image(ref)))
        t = extract_features(segment_signature(preprocess_image(test)))

        sim = 1.0 - cosine(r, t)

        if np.isnan(sim):
            return 0.0

        return float(np.clip(sim, 0.0, 1.0))

    except Exception as e:
        print(f"[WARN] HOG similarity failed: ref={ref}, test={test}, err={e}")
        return 0.0
