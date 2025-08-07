import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# ✅ Set page config as the very first Streamlit command  
st.set_page_config(page_title="Decoding Book Covers", layout="centered")

# --- Sidebar: Model Selector ---
st.sidebar.title("🔧 Model Selection")
model_options = {
    "EfficientNetB0 (Default)": "Saved_Model/EfficientNetB0_books3.keras",
    "ResNet50": "Saved_Model/ResNet50_books3.keras",
    "ImageNet Custom": "Saved_Model/ImageNet_books3.keras"
}

# Dropdown list for model selection
selected_model_name = st.sidebar.selectbox("Choose a model:", list(model_options.keys()), index=0)
selected_model_path = model_options[selected_model_name]

# Load the selected model 
@st.cache_resource(show_spinner="Loading model...")
def load_selected_model(model_path):
    return load_model(model_path)

model = load_selected_model(selected_model_path)

# --- Class labels ---
class_indices = {
    'Children': 0,
    'History': 1,
    'Romance': 2,
    'Sport': 3,
    'Unknown': 4,
}
labels = {v: k for k, v in class_indices.items()}

# --- Main UI ---
st.title("📚 Decoding Book Covers")
st.write("Upload a book cover image and click **Predict Label** to classify it.")

uploaded_file = st.file_uploader("Upload a book cover image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, width=50)

    if st.button("🔍 Predict Label"):
        # Preprocess image
        img = image.resize((224, 224))  # Make sure this matches model input size!
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(predictions[0][predicted_index]) * 100

        # Handle missing label gracefully
        predicted_label = labels.get(predicted_index, f"Unknown (class {predicted_index})")

        st.success(f"✅ Predicted class: **{predicted_label}** ({confidence:.2f}%)")

        # Show probability bar chart
        st.subheader("📊 Prediction Probabilities")
        prob_dict = {
            labels.get(i, f"Unknown {i}"): float(prob * 100)
            for i, prob in enumerate(predictions[0])
        }
        st.bar_chart(prob_dict)
