import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # For image handling

# Load TensorFlow Model Once
import os

@st.cache_resource
def load_model():
model_path = "my_model.h5"  # Update if in a subdirectory

if os.path.exists(model_path):
    print(f"✅ Model file found at: {os.path.abspath(model_path)}")
else:
    print("❌ Model file NOT found. Check the path and permissions.")



model = load_model()

# Define Class Names (Updated)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Model Prediction Function
def model_prediction(test_image):
    try:
        image = Image.open(test_image).convert("RGB")  # Convert image to RGB
        image = image.resize((128, 128))  # Resize to model input size
        input_arr = np.array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        
        st.write(f"🔍 Processed Image Shape: {input_arr.shape}")  # Debugging
        st.image(image, caption="Preprocessed Image", use_column_width=True)

        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        
        st.write(f"🔍 Raw Prediction Output: {prediction}")  # Debugging
        return result_index
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
        return None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help identify plant diseases efficiently. Upload an image, and our system will analyze it.
    
    ### How It Works:
    1. **Upload Image** (Disease Recognition Page)
    2. **Analysis** (AI processes the image)
    3. **Results** (View disease classification)
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 classes.
    
    #### Dataset Details:
    - **Train:** 70,295 images
    - **Validation:** 17,572 images
    - **Test:** 33 images
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Please wait..."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    st.success(f"Model Prediction: **{class_names[result_index]}**")
