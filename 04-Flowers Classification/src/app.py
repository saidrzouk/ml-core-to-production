import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

from src.config import CLASS_NAMES_PATH, IMG_SIZE, MODEL_PATH


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return {int(index): name for name, index in data.items()}


def preprocess_image(uploaded_file):
    image = load_img(uploaded_file, target_size=IMG_SIZE)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return array


st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("Flower Classifier")
st.write("Upload a flower image and the model will predict the class.")

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    image = preprocess_image(uploaded_file)
    prediction = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(prediction))

    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: {confidence:.2%}")

    st.bar_chart({class_names[i]: float(prob) for i, prob in enumerate(prediction)})
