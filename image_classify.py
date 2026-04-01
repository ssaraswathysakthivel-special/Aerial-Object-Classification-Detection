import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Aerial Image Classifier", layout="wide")
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 10px;
    ">
        <h1 style="color: white; font-size:2.5rem; margin:0;">
        🛰️ Aerial Image Classification
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.subheader(":red[Upload an image to classify: **Bird 🐦 or  Drone 🚁**]")

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r"C:\Users\ADMIN\Documents\mini_project_guvi\project_aerial_object_classification&detection\best_model.keras",
        custom_objects={'preprocess_input': preprocess_input}
    )
    return model
model = load_model()

IMG_SIZE = 128
# -----------------------
# Image Upload
# -----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------
    # Preprocess Image
    # -----------------------
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # -----------------------
    # Prediction
    # -----------------------
    prediction = model.predict(img_array)[0][0]
    st.write(prediction)

    if prediction > 0.5:
        label = "Drone 🚁"
        confidence = prediction
    else:
        label = "Bird 🐦"
        confidence = 1 - prediction

    # -----------------------
    # Display Result
    # -----------------------
    st.subheader("Prediction:")
    st.success(f"{label}")
    st.subheader(f"Confidence: **{confidence*100:.2f}%**")