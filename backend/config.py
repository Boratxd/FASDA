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

# PostgreSQL connection settings (AWS RDS)
PG_HOST = "fasda-db.c9acs4wi65jb.eu-north-1.rds.amazonaws.com"
PG_PORT = 5432
PG_DATABASE = "fasda"
PG_USER = "fasdaPostgres"
PG_PASSWORD = "fasda2526"

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

IMG_SIZE = (128, 128)

ALPHA = 0.4 # Alpha value - Weight
THRESHOLD = 0.55 # This is our threshold value
