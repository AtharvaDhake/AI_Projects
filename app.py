import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Ensure model file exists
MODEL_PATH = "my_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found. Please check the file path.")

# Load TensorFlow Model
def load_model():
    try:
        st.write("🔍 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    try:
        image = Image.open(test_image).convert("RGB")  # Convert to RGB
        image = image.resize((128, 128))  # Resize to match model input
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
        return None

# Sidebar
st.sidebar.title("🌿 Plant Disease Dashboard")
app_mode = st.sidebar.selectbox("📌 Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌿🔍  

    **How It Works:**  
    1. 📷 **Upload an image** of a plant with suspected diseases.  
    2. 🤖 **AI analyzes** the image to detect potential diseases.  
    3. 📊 **View the results** and get recommendations.  
    """)

# About Page
elif app_mode == "About":
    st.header("📊 About the Dataset")
    st.markdown("""
    This dataset contains **87,000+ RGB images** of healthy and diseased crop leaves, categorized into **38 classes**.

    - **Train:** 70,295 images  
    - **Validation:** 17,572 images  
    - **Test:** 33 images  
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")
    test_image = st.file_uploader("📂 Upload an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("🔍 Analyzing... Please wait..."):
                if model:
                    result_index = model_prediction(test_image)
                    if result_index is not None:
                        class_names = [
                            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                        ]
                        st.success(f"🌿 The model predicts: **{class_names[result_index]}**")
                    else:
                        st.error("⚠️ Error making prediction.")
                else:
                    st.error("⚠️ Model not loaded. Please check the model file.")
