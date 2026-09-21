import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "final_model_under_25mb_float16.keras",
    compile=False
)

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

st.title("Waste Classification")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    # Resize image to model input size
    image_resized = image.resize((224, 224))

    # Convert to NumPy array
    img_array = np.array(image_resized)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    st.success(f"Predicted Waste: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
