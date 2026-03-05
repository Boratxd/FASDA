# ✍️ FASDA — Forgery Attendance Signature Detection App  
**A Machine-Learning–powered system for detecting forged signatures on university attendance sheets.**

FASDA is a full-stack application designed to help instructors automatically verify the authenticity of signatures collected during in-class attendance.  
The system processes attendance sheet uploads, segments student signatures, runs them through a machine learning pipeline, and reports potential forgeries with high accuracy.

This project was developed as part of the **CNG491 Graduation Project** at **Middle East Technical University, Northern Cyprus Campus (METU NCC)**.

---

## 🚀 Features

### 🔍 Automated Signature Forgery Detection
- Upload attendance sheets (PDF / JPG / PNG)
- Automatic segmentation of signatures from the sheet
- Preprocessing (binarization, thresholding, normalization)
- ML-based comparison using a **Siamese Neural Network**

### 📊 Real-time Results & Visualization
- Genuine / Forged / Irregular signatures highlighted with colors
- Per-signature confidence scores
- Detailed breakdown per student

### 🧑‍🏫 Instructor Tools
- Instructor dashboard
- Manual flagging of suspicious signatures
- Downloadable verification reports (PDF/JSON)

### 🔗 External Integrations
- **SheerID** verification for instructor identity
- **Bank API** for secure activation payments

---

## 🏗️ System Architecture

### The system consists of three main components:

1. **Frontend (React + TailwindCSS)**  
   - Upload interface  
   - Signature visualization  
   - Flagging dashboard  
   - Report viewer  

2. **Backend API (FastAPI)**  
   - File upload and validation  
   - Preprocessing service  
   - ML inference  
   - User authentication & SheerID mock  
   - Report generation  

3. **ML Pipeline (Python / PyTorch)**  
   - Signature segmentation  
   - Preprocessing  
   - Siamese network embedding generation  
   - Similarity scoring  

4. **Database (PostgreSQL)**  
   - Users  
   - Attendance sheets  
   - Signatures  
   - Classification results  
   - Flags  

---

## 🧪 Machine Learning Model

### Model Architecture
- Siamese Convolutional Neural Network  
- Contrastive Loss  
- Feature embedding size: 128-d  
- Threshold-based classification

### Training Pipeline
- Dataset: GPDS / CEDAR (licensed usage)  
- Preprocessing steps:  
  - Grayscale  
  - Gaussian blur  
  - Otsu threshold  
  - Contour extraction  
  - Resize to 128x128  

### Output  
The model returns:

```json
{
  "signature_id": "1245",
  "prediction": "forged",
  "confidence": 0.82
}
