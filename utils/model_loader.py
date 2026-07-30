from tensorflow.keras.models import load_model
import joblib
import streamlit as st

@st.cache_resource
def load_cnn_model():

    model = load_model("models/wafer_defect_cnn.keras")

    print("\n===== MODEL LOADED BY STREAMLIT =====")
    for layer in model.layers:
        print(layer.name)
    print("=====================================\n")

    encoder = joblib.load("models/label_encoder.pkl")

    return model, encoder