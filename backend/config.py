import os

# =========================================================
# CONFIGURATION
# =========================================================
DATASET_PATH = r"C:\Users\alper\Desktop\CNG 491\DataSet\SignatureGPDSSyntheticSignaturesManuscriptsv\firmasSINTESISmanuscritas"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gpds_siamese_modelv4.h5"
)

# SQLite DB location:
DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "fasda_results.db"
)

IMG_SIZE = (128, 128)

ALPHA = 0.4 # Alpha value - Weight
THRESHOLD = 0.55 # This is our threshold value
