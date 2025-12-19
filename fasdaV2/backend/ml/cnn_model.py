# ml/cnn_model.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from config import IMG_SIZE

# =========================================================
# SAFE DISTANCE LAYER
# =========================================================
class L1DistanceLayer(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

# =========================================================
# IMAGE PREPROCESSING (CNN)
# =========================================================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    img = img / 255.0
    return img.reshape(128, 128, 1)

# =========================================================
# SIAMESE CNN
# =========================================================
def build_feature_extractor():
    inp = layers.Input((128,128,1))
    x = layers.Conv2D(64,3,activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3,activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    return Model(inp, layers.Dense(64)(x))

def build_siamese_model():
    base = build_feature_extractor()
    a = layers.Input((128,128,1))
    b = layers.Input((128,128,1))
    d = L1DistanceLayer()([base(a), base(b)])
    out = layers.Dense(1, activation="sigmoid")(d)
    model = Model([a,b], out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model
