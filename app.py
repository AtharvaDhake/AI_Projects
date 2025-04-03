import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# Load model with caching
@st.cache_resource
def load_model():
    model_path = "my_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class names
CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Cercospora Spot', 'Corn Common Rust', 'Corn Northern Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy',
    'Orange Citrus Greening', 'Peach Bacterial Spot', 'Peach Healthy',
    'Bell Pepper Bacterial Spot', 'Bell Pepper Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Spot', 'Tomato Spider Mites',
    'Tomato Target Spot', 'Tomato Yellow Curl Virus', 'Tomato Mosaic Virus',
    'Tomato Healthy'
]

# Preprocess and predict
def predict_disease(image, model):
    try:
        img = Image.open(image).convert('RGB')
        img = img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        return CLASS_NAMES[predicted_class], confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Disease Detection"])

# Home Page
if page == "Home":
    st.title("🌱 Plant Disease Recognition System")
    st.image("home_page.jpg", use_column_width=True)
    st.markdown("""
    Welcome to our Plant Disease Recognition System! Upload an image of a plant leaf, 
    and our AI will help identify potential diseases.
    
    **How to use:**
    1. Go to the **Disease Detection** page
    2. Upload an image of a plant leaf
    3. Click 'Predict' to analyze the image
    """)

# About Page
elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    **About the Dataset:**
    - Contains 87,000+ images of healthy and diseased plant leaves
    - Covers 38 different plant disease categories
    - Divided into training (70,295), validation (17,572), and test (33) sets
    
    **Model Information:**
    - Uses a deep learning model trained on the PlantVillage dataset
    - Achieves high accuracy in identifying common plant diseases
    """)

# Detection Page
elif page == "Disease Detection":
    st.title("🔍 Disease Detection")
    model = load_model()
    
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict Disease"):
            if model is None:
                st.error("Model not loaded. Cannot make predictions.")
            else:
                with st.spinner("Analyzing..."):
                    disease, confidence = predict_disease(uploaded_file, model)
                    
                    if disease:
                        with col2:
                            st.success(f"**Prediction:** {disease}")
                            st.info(f"**Confidence:** {confidence:.2f}%")
                            
                            # Show treatment suggestions for diseases (not healthy plants)
                            if "Healthy" not in disease:
                                st.warning("**Treatment Suggestions:** Consider using appropriate fungicides or pesticides. Remove infected leaves to prevent spread.")
                            else:
                                st.success("**Status:** Plant appears healthy!")
