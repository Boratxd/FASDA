# ml/dataset_utils.py
import os

# =========================================================
# DATASET HANDLING
# =========================================================
def extract_user_signatures(user_dir):
    files = os.listdir(user_dir)
    genuine = [f for f in files if f.lower().startswith("c-")]
    forged  = [f for f in files if f.lower().startswith("cf-")]
    return genuine, forged
