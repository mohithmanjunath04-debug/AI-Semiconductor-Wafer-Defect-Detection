from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.gradcam import generate_gradcam, overlay_heatmap
from skimage.transform import resize
from utils.pdf_generator import generate_pdf

from utils.model_loader import load_cnn_model
from utils.data_loader import load_dataset

# ==========================================
# Page Configuration
# ========================================
# ==========================================
# Load Model & Dataset
# ==========================================
model, encoder = load_cnn_model()
df, df_clean = load_dataset()

st.set_page_config(
    page_title="AI Wafer Defect Detector",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=90
    )

    st.title("🤖 AI Wafer Inspector")

    st.markdown("---")

    st.success("✅ CNN Model Loaded")

    st.metric(
        "📂 Dataset Size",
        len(df_clean)
    )

    st.metric(
        "🏷️ Defect Classes",
        len(encoder.classes_)
    )

    st.metric(
        "🧠 CNN Input Size",
        "32 × 32"
    )

    st.markdown("---")

    st.write("### 📌 Project")

    st.info("""
**Technology Stack**

• Python

• TensorFlow / Keras

• CNN

• Streamlit

• LSWMD Dataset
""")

    st.markdown("---")

    st.write("### 📌 Version")
    st.write("v1.0")

    st.write("### 👨‍💻 Developed By")
    st.write("Mohith Manjunath")

    st.markdown("---")

    st.caption(
        "AI Semiconductor Wafer Defect Detection using CNN + Explainable AI"
    )

# ==========================================
# Title
# ==========================================

st.title("🧠 AI Semiconductor Wafer Defect Detection")

st.markdown("""
### 🚀 CNN-Based Intelligent Wafer Inspection System

This dashboard performs automated semiconductor wafer defect classification
using a Convolutional Neural Network (CNN) trained on the LSWMD dataset.

It also integrates **Explainable AI (Grad-CAM)** to visualise the regions
that influenced the model's prediction.

---
""")

st.markdown("""
### Deep Learning based Wafer Inspection System

This application detects semiconductor wafer defects using a trained
Convolutional Neural Network (CNN).

---
""")

st.info("""
🎯 **Project Objective**

This application uses a Convolutional Neural Network (CNN) trained on the
LSWMD semiconductor wafer dataset to automatically identify wafer defect
patterns.

The goal is to assist semiconductor manufacturers in improving yield,
reducing inspection time, and minimizing production costs.
""")

# ==========================================
# Load Model
# ==========================================

model, encoder = load_cnn_model()
print(type(model))
print(model.inputs)
print(model.outputs)

class_names = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch"
]

st.markdown("---")

st.header("📈 Dataset Statistics")

col1, col2, col3 = st.columns(3)

df, df_clean = load_dataset()

with col1:
    st.metric(
        "📦 Labelled Wafers",
        f"{len(df_clean):,}"
    )

with col2:
    st.metric(
        "🧩 Image Size",
        "32 × 32"
    )

with col3:
    st.metric(
        "🧠 CNN Classes",
        len(encoder.classes_)
    )

# ==========================================
# Load Dataset
# ==========================================

df, df_clean = load_dataset()
# ==========================================
# Load Evaluation Files
# ==========================================

report_df = pd.read_csv("models/classification_report.csv", index_col=0)

conf_matrix = np.load("models/confusion_matrix.npy")
# ==========================================
# Load Overall Evaluation Metrics
# ==========================================

accuracy = report_df.loc["accuracy", "precision"]

precision = report_df.loc["weighted avg", "precision"]

recall = report_df.loc["weighted avg", "recall"]

f1 = report_df.loc["weighted avg", "f1-score"]
# ==========================================
# Dataset Information
# ==========================================

st.success("✅ CNN Model Loaded Successfully!")
st.metric(
    label="🎯 CNN Test Accuracy",
    value=f"{accuracy*100:.2f}%"
)

st.header("📊 Dataset Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "📦 Total Wafers",
        f"{len(df):,}"
    )

with c2:
    st.metric(
        "🎯 Defect Samples",
        f"{len(df_clean):,}"
    )

with c3:
    st.metric(
        "🧠 CNN Classes",
        len(class_names)
    )

with c4:
    st.metric(
    "🎯 Test Accuracy",
    f"{accuracy*100:.2f}%"
)

# Wafer Explorer
# ==========================================

st.markdown("---")
st.markdown("---")

st.header("📚 Defect Classes")

legend = {
    "Center": "Defect at wafer centre",
    "Donut": "Circular defect pattern",
    "Edge-Loc": "Localized defect near the wafer edge",
    "Edge-Ring": "Ring-shaped defect around the wafer edge",
    "Loc": "Localized defect",
    "Near-full": "Almost the entire wafer is defective",
    "Random": "Randomly distributed defects",
    "Scratch": "Scratch-like defect"
}

st.table(
    pd.DataFrame(
        legend.items(),
        columns=["Defect", "Description"]
    )
)

st.header("🔎 Wafer Explorer")

st.write(
    "Select any wafer from the dataset to compare the **actual defect** with the **CNN prediction**."
)

wafer_index = st.slider(
    "🔍 Select a Wafer Sample",
    min_value=0,
    max_value=len(df_clean) - 1,
    value=0,
    step=1
)

st.caption(f"Currently Viewing Wafer #{wafer_index}")
# -----------------------------------
# Get Selected Wafer
# -----------------------------------
# ==========================================
# Get Selected Wafer
# ==========================================

wafer = df_clean.iloc[wafer_index]["waferMap"]

# Resize wafer for CNN
image = resize(
    wafer,
    (32, 32),
    preserve_range=True,
    anti_aliasing=False
)

image = image.reshape(1, 32, 32, 1)

# CNN Prediction
prediction = model.predict(image, verbose=0)
heatmap = generate_gradcam(
    model,
    image,
    "last_conv"
)

overlay = overlay_heatmap(
    heatmap,
    wafer
)

predicted_class = prediction.argmax()

confidence = float(prediction[0][predicted_class]) * 100

predicted_label = encoder.inverse_transform([predicted_class])[0]
# Actual Label
actual_label = df_clean.iloc[wafer_index]["failureLabel"]
st.markdown("## 📊 Prediction Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Prediction",
        predicted_label
    )

with col2:
    st.metric(
        "📈 Confidence",
        f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        "⚠ Actual Label",
        actual_label
    )

with col4:
    st.metric(
        "🤖 Model",
        "CNN v1.0"
    )

# Prediction Status
if predicted_label == actual_label:
    st.success("✅ Correct Prediction")
else:
    st.error("❌ Incorrect Prediction")

# Model Accuracy
st.info(
    "📌 Model Validation Accuracy: 88.38%"
)
st.markdown("### 🎯 Prediction Confidence")

st.progress(confidence / 100)

st.write(f"**Confidence Score:** {confidence:.2f}%")
st.markdown("## 📈 Project Statistics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "📂 Total Wafers",
        len(df_clean)
    )

with c2:
    st.metric(
        "🏷️ Defect Classes",
        len(encoder.classes_)
    )

with c3:
    st.metric(
        "🧠 CNN Input Size",
        "32 × 32"
    )

st.markdown("---")

# Create probability dataframestre
prob_df = pd.DataFrame({
    "Defect": encoder.classes_,
    "Probability (%)": prediction[0] * 100
})

prob_df = prob_df.sort_values(
    by="Probability (%)",
    ascending=False
)

# Sort from highest probability

# ==========================================
# Show Wafer Image
# ==========================================

fig, ax = plt.subplots(figsize=(6,6))

ax.imshow(
    wafer,
    cmap="viridis"
)

ax.set_title(
    f"Selected Wafer #{wafer_index}",
    fontsize=15,
    fontweight="bold"
)

ax.axis("off")

ax.imshow(wafer, cmap="gray")
ax.set_title("Selected Wafer")
ax.axis("off")

# ==========================================
# Display Results
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="🏷 Actual Defect",
        value=actual_label
    )

with col2:
    st.metric(
        label="🤖 CNN Prediction",
        value=predicted_label
    )

    st.subheader("🎯 Confidence")

    if confidence > 90:
        st.success(f"{confidence:.2f}%")

    elif confidence > 70:
        st.warning(f"{confidence:.2f}%")

    else:
        st.error(f"{confidence:.2f}%")

    st.progress(float(confidence) / 100.0)

    st.caption(
        f"🕒 Prediction generated on {datetime.now().strftime('%d %b %Y, %H:%M:%S')}"
    )
# ==========================================
# Display
# ==========================================
if actual_label == predicted_label:
    st.balloons()
    st.success("✅ Excellent! CNN Prediction Matches Ground Truth")
else:
    st.warning("⚠ CNN Prediction Does Not Match Ground Truth")

fig, ax = plt.subplots(figsize=(6,6))

ax.imshow(wafer, cmap="inferno")

ax.axis("off")

st.pyplot(fig)
st.markdown("---")
st.markdown("## 🧠 Explainable AI")
st.subheader("🧠 Explainable AI (Grad-CAM)")

col1, col2, col3 = st.columns(3)

with col1:

    display_wafer = wafer.astype("float32")

    display_wafer = display_wafer / display_wafer.max()

    st.image(
        display_wafer,
        caption="Original Wafer",
        clamp=True,
        use_container_width=True
    )

with col2:

    display_heatmap = resize(
        heatmap,
        wafer.shape,
        preserve_range=True,
        anti_aliasing=True
    )

    st.image(
        display_heatmap,
        caption="Grad-CAM Heatmap",
        clamp=True,
        use_container_width=True
    )

with col3:
    st.image(
        overlay,
        caption="Overlay",
        use_container_width=True
    )
    st.info("""
🔍 **Grad-CAM Interpretation**

The highlighted regions indicate the areas of the wafer that most influenced
the CNN's prediction.

- 🔴 Red = High importance
- 🟡 Yellow = Moderate importance
- 🔵 Blue = Low importance

This visualization improves the transparency and interpretability of the AI model.
""")
st.markdown("---")

st.subheader("📊 Top 3 Predictions")
top3 = prob_df.head(3).copy()

medals = ["#1", "#2", "#3"]

top3["Rank"] = medals

top3 = top3[["Defect", "Probability (%)"]]

st.dataframe(
    top3,
    use_container_width=True,
    hide_index=True
)
# ==========================================
# Generate PDF Report
# ==========================================

top3 = prob_df.sort_values(
    "Probability (%)",
    ascending=False
).head(3)

pdf_file = generate_pdf(
    wafer_no=wafer_index,
    actual_label=actual_label,
    predicted_label=predicted_label,
    confidence=confidence,
    accuracy=accuracy * 100,
    precision=precision * 100,
    recall=recall * 100,
    f1=f1 * 100,
    top3=top3
)
with open(pdf_file, "rb") as pdf:

    st.download_button(
        label="📄 Download AI Prediction Report",
        data=pdf,
        file_name=f"Wafer_{wafer_index}_Prediction_Report.pdf",
        mime="application/pdf"
    )
import plotly.express as px

st.markdown("---")
st.header("📊 Prediction Confidence")

chart_df = prob_df.head(3)

fig = px.bar(
    chart_df,
    x="Probability (%)",
    y="Defect",
    orientation="h",
    text="Probability (%)",
    color="Probability (%)",
    color_continuous_scale="Blues"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Confidence (%)",
    yaxis_title="",
    yaxis=dict(categoryorder="total ascending"),
    height=350,
    coloraxis_showscale=False,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.header("📈 Model Evaluation")
accuracy = report_df.loc["accuracy"][0]

accuracy = report_df.loc["accuracy"][0]

precision = report_df.loc["weighted avg"]["precision"]

recall = report_df.loc["weighted avg"]["recall"]

f1 = report_df.loc["weighted avg"]["f1-score"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")

with col2:
    st.metric("📌 Precision", f"{precision*100:.2f}%")

with col3:
    st.metric("📈 Recall", f"{recall*100:.2f}%")

with col4:
    st.metric("⭐ F1 Score", f"{f1*100:.2f}%")


st.subheader("📋 Classification Report")

st.dataframe(
    report_df.style.background_gradient(cmap="Greens")
)

class_report = report_df.iloc[:-3]

best_class = class_report["f1-score"].idxmax()
best_score = class_report["f1-score"].max()

st.success(
    f"🏆 Best Performing Class: {best_class} (F1 Score: {best_score:.2f})"
)

worst_class = class_report["f1-score"].idxmin()
worst_score = class_report["f1-score"].min()

st.warning(
    f"⚠ Lowest Performing Class: {worst_class} (F1 Score: {worst_score:.2f})"
)
st.subheader("📊 Confusion Matrix")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.divider()

st.caption(
    "🧠 AI Semiconductor Wafer Defect Detection | CNN + TensorFlow | Developed by Mohith Manjunath"
)
st.markdown("---")

st.header("📘 About This Project")

st.write("""
This AI-powered Semiconductor Wafer Defect Detection System uses a
Convolutional Neural Network (CNN) to automatically classify
semiconductor wafer defects into eight different defect categories.

The application also integrates Explainable AI (Grad-CAM),
allowing users to visualize which regions of the wafer
influenced the CNN's prediction.

### 🔍 Features

✅ CNN-based wafer defect classification

✅ Explainable AI (Grad-CAM)

✅ Top-3 prediction probabilities

✅ Interactive Streamlit dashboard

✅ Confusion Matrix

✅ Classification Report

### 🎯 Objective

To assist semiconductor manufacturers in detecting wafer defects
accurately, efficiently, and transparently using Artificial Intelligence.
""")

st.markdown("---")

st.caption(
    "© 2026 Mohith Manjunath | AI Semiconductor Wafer Defect Detection"
)