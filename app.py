import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# --------------------------------------------------
# LANGUAGE TEXT
# --------------------------------------------------

languages = {
    "English": {
        "title": "♻️ Waste Classification Using CNN",
        "subtitle": "Upload an image to identify the type of waste",
        "upload": "Upload a waste image",
        "uploaded": "Uploaded Image",
        "prediction": "Prediction",
        "waste_type": "Waste Type",
        "confidence": "Confidence",
        "parent": "Waste Category",
        "hazardous": "Hazardous",
        "biodegradable": "Biodegradable",
        "non_biodegradable": "Non-Biodegradable",
        "analyzing": "Analyzing image...",
        "error": "Unable to process this image."
    },

    "தமிழ்": {
        "title": "♻️ CNN பயன்படுத்தி கழிவு வகைப்படுத்துதல்",
        "subtitle": "கழிவின் வகையை கண்டறிய படத்தை பதிவேற்றவும்",
        "upload": "கழிவின் படத்தை பதிவேற்றவும்",
        "uploaded": "பதிவேற்றப்பட்ட படம்",
        "prediction": "கணிப்பு",
        "waste_type": "கழிவு வகை",
        "confidence": "நம்பகத்தன்மை",
        "parent": "கழிவு வகைப்பாடு",
        "hazardous": "அபாயகரமான கழிவு",
        "biodegradable": "மக்கும் கழிவு",
        "non_biodegradable": "மக்காத கழிவு",
        "analyzing": "படத்தை ஆய்வு செய்கிறது...",
        "error": "இந்த படத்தை செயல்படுத்த முடியவில்லை."
    },

    "हिन्दी": {
        "title": "♻️ CNN का उपयोग करके कचरा वर्गीकरण",
        "subtitle": "कचरे के प्रकार की पहचान करने के लिए एक तस्वीर अपलोड करें",
        "upload": "कचरे की तस्वीर अपलोड करें",
        "uploaded": "अपलोड की गई तस्वीर",
        "prediction": "पूर्वानुमान",
        "waste_type": "कचरे का प्रकार",
        "confidence": "विश्वसनीयता",
        "parent": "कचरे की श्रेणी",
        "hazardous": "खतरनाक कचरा",
        "biodegradable": "जैव-अवक्रमणीय कचरा",
        "non_biodegradable": "गैर-जैव-अवक्रमणीय कचरा",
        "analyzing": "तस्वीर का विश्लेषण किया जा रहा है...",
        "error": "इस तस्वीर को संसाधित नहीं किया जा सका।"
    }
}

# --------------------------------------------------
# LANGUAGE SELECTION
# --------------------------------------------------

language = st.selectbox(
    "🌐 Language / மொழி / भाषा",
    ["English", "தமிழ்", "हिन्दी"]
)

text = languages[language]

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(text["title"])
st.write(text["subtitle"])

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_model_under_25mb_float16.keras"
    )

@st.cache_resource
def load_class_names():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    class_names = load_class_names()

except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# WASTE CATEGORY MAPPING
# --------------------------------------------------

hazardous = [
    "Battery",
    "Paints",
    "Pesticides"
]

biodegradable = [
    "Diapers",
    "Sanitary_napkin",
    "Textile Trash"
]

non_biodegradable = [
    "Cans",
    "Cardboard",
    "Ceramic_product",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Plastic_bottles",
    "Platics_bags_wrappers",
    "Shoes",
    "Styrofoam_product"
]

def get_parent_category(waste):

    if waste in hazardous:
        return text["hazardous"]

    elif waste in biodegradable:
        return text["biodegradable"]

    else:
        return text["non_biodegradable"]


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    text["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption=text["uploaded"],
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert to NumPy array
    img_array = np.array(image_resized)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    with st.spinner(text["analyzing"]):

        try:
            predictions = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(predictions[0])

            predicted_class = class_names[predicted_index]

            confidence = (
                float(predictions[0][predicted_index]) * 100
            )

        except Exception:
            st.error(text["error"])
            st.stop()

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader(text["prediction"])

    st.success(
        f"**{text['waste_type']}:** {predicted_class}"
    )

    st.info(
        f"**{text['confidence']}:** {confidence:.2f}%"
    )

    parent_category = get_parent_category(
        predicted_class
    )

    st.write(
        f"**{text['parent']}:** {parent_category}"
    )
