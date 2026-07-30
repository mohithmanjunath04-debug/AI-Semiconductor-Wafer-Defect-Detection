# 🧠 AI Semiconductor Wafer Defect Detection using CNN & Explainable AI

## 📌 Project Overview

This project presents an AI-powered semiconductor wafer defect detection system using a Convolutional Neural Network (CNN) trained on the **LSWMD (Large-Scale Wafer Map Dataset)**.

The system automatically classifies wafer defects into multiple defect categories and integrates **Grad-CAM (Explainable AI)** to visualize the regions that influenced the model's predictions.

---

## 🚀 Features

- ✅ CNN-based wafer defect classification
- ✅ Explainable AI using Grad-CAM
- ✅ Top-3 prediction probabilities
- ✅ Confidence score
- ✅ Actual vs Predicted comparison
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ PDF Prediction Report
- ✅ Interactive Streamlit Dashboard

---

## 🛠 Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- ReportLab
- Scikit-Learn

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **87.66%** |
| Precision | **88.87%** |
| Recall | **87.66%** |
| F1 Score | **87.97%** |

---

## 🖼 Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

---

### Prediction

![Prediction](screenshots/prediction.png)

---

### Explainable AI (Grad-CAM)

![GradCAM](screenshots/gradcam.png)

---

### Model Evaluation

![Evaluation](screenshots/evaluation.png)

---

### PDF Report

![PDF](screenshots/pdf_report.png)

---

## 📂 Project Structure

```text
AI-Semiconductor-Wafer-Defect-Detection/

├── app.py
├── README.md
├── requirements.txt
├── models/
├── notebooks/
├── utils/
├── screenshots/
└── data/
```

---

## ▶️ How to Run

```bash
git clone <repository>

cd AI-Semiconductor-Wafer-Defect-Detection

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 📚 Dataset

**LSWMD – Large Scale Wafer Map Dataset**

The dataset contains semiconductor wafer maps categorized into multiple defect classes and is widely used for AI-based semiconductor defect analysis.

---

## 🔬 Explainable AI

Grad-CAM is integrated to improve model interpretability by highlighting the regions of the wafer that contribute most to the CNN's prediction.

---

## 👨‍💻 Developer

**Mohith Manjunath**

Computer Science & Engineering

AI • Machine Learning • Semiconductor Manufacturing

---

## 📄 License

This project is developed for academic and research purposes.