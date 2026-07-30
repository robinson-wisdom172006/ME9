"""
GET 324 Mini-Project (Group ME9)
Streamlit Web App: Tomato Leaf Mold vs Tomato Septoria Leaf Spot Classifier

Run locally with:   streamlit run app.py

Deploy for free on Streamlit Community Cloud:
1. Push this whole folder to a GitHub repository (must include tomato_model.h5,
   class_names.txt, app.py, requirements.txt).
2. Go to https://share.streamlit.io , sign in with GitHub.
3. Click "New app", pick your repo/branch, set Main file path = app.py.
4. Click Deploy. You'll get a public URL to submit.

NOTE: tomato_model.h5 can be large. If GitHub rejects the push (>100MB),
use Git LFS (see README.md) or reduce model size.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="centered",
)

MODEL_PATH = "tomato_model.h5"
CLASS_NAMES_FILE = "class_names.txt"
IMG_SIZE = (224, 224)


# ----------------------------
# LOAD MODEL (cached so it only loads once per session)
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def readable_label(name: str) -> str:
    return name.replace("_", " ").replace("Tomato ", "Tomato: ")


# ----------------------------
# UI
# ----------------------------
st.title("🍅 Tomato Leaf Disease Classifier")
st.write(
    "Upload a photo of a tomato leaf to classify it as either "
    "**Tomato Leaf Mold** or **Tomato Septoria Leaf Spot**."
)

model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Choose a leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]  # sigmoid output, 0-1

        # class_names[0] corresponds to model output 0, class_names[1] to output 1
        if prediction >= 0.5:
            predicted_class = class_names[1]
            confidence = prediction
        else:
            predicted_class = class_names[0]
            confidence = 1 - prediction

    st.success(f"**Prediction:** {readable_label(predicted_class)}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.progress(float(confidence))

    with st.expander("See raw model output"):
        st.write(f"Raw sigmoid score: {prediction:.4f}")
        st.write(f"Class mapping: {dict(enumerate(class_names))}")

st.markdown("---")
st.caption(
    "GET 324 Mini-Project · Group ME9 · Mechanical Engineering · "
    "Model: MobileNetV2 (transfer learning)"
)
