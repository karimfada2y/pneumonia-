import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# Google Drive model download
MODEL_PATH = "pneumonia_model.h5"
FILE_ID = "1iwWCz2DIu9dSPHTAbE06EIuFjnADpnJl"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# Streamlit UI
st.title("🩻 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict if it's NORMAL or shows PNEUMONIA.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🧠 Prediction: **{result}**")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
