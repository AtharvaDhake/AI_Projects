import subprocess
import sys

# Force-install dependencies if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

REQUIRED_PACKAGES = [
    ("tensorflow-cpu", "2.15.0"),
    ("Pillow", "10.1.0"),
    ("numpy", "1.23.5"),
    ("protobuf", "3.20.3")
]

for pkg, ver in REQUIRED_PACKAGES:
    try:
        __import__(pkg.split("[")[0])
    except ImportError:
        install(f"{pkg}=={ver}")

import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌱",
    layout="wide"
)

# Load TensorFlow Model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('my_model.keras')
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# TensorFlow Model Prediction Function with enhanced error handling
def model_prediction(test_image):
    try:
        image = Image.open(test_image).convert("RGB")
        image = image.resize((128, 128))
        input_arr = np.array(image) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)
        
        # Check if model is loaded
        if model is None:
            st.error("Model not loaded properly. Cannot make predictions.")
            return None
            
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return result_index, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Class names
CLASS_NAMES = [
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

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Page", 
                           ["Home", "Disease Detection", "About"],
                           index=0)

# Home Page
if app_mode == "Home":
    st.title("🌿 Plant Disease Recognition System")
    st.markdown("""
    ### Welcome to Our Plant Health Diagnostic Tool!
    
    Our advanced AI system helps farmers, gardeners, and agricultural professionals 
    identify plant diseases quickly and accurately. Upload an image of a plant leaf, 
    and our system will analyze it to detect any signs of diseases.
    """)
    
    # Columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ce", 
                caption="Healthy crops lead to better harvests", use_column_width=True)
    
    with col2:
        st.markdown("""
        #### Why Early Detection Matters:
        - Prevents spread of disease to other plants
        - Reduces crop losses
        - Minimizes pesticide use through targeted treatment
        - Increases overall yield quality and quantity
        
        #### How It Works:
        1. Navigate to **Disease Detection** page
        2. Upload a clear image of a plant leaf
        3. Get instant diagnosis and recommendations
        """)
    
    st.warning("⚠️ Note: This tool provides preliminary diagnosis only. For critical cases, consult with agricultural experts.")

# Disease Detection Page
elif app_mode == "Disease Detection":
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload an image of a plant leaf to check for potential diseases.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Select a clear image of a plant leaf")
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Prediction button
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    result_index, confidence = model_prediction(uploaded_file)
                    
                    if result_index is not None and confidence is not None:
                        disease = CLASS_NAMES[result_index]
                        plant = disease.split("___")[0].replace("_", " ")
                        status = disease.split("___")[1].replace("_", " ")
                        
                        # Display results
                        with col2:
                            st.subheader("Analysis Results")
                            
                            if "healthy" in status.lower():
                                st.success(f"**Plant:** {plant}")
                                st.success(f"**Status:** {status}")
                                st.success(f"**Confidence:** {confidence:.2f}%")
                                st.balloons()
                                st.markdown("🎉 Great news! Your plant appears to be healthy!")
                            else:
                                st.warning(f"**Plant:** {plant}")
                                st.error(f"**Disease Detected:** {status}")
                                st.info(f"**Confidence:** {confidence:.2f}%")
                                
                                # Basic recommendations
                                st.markdown("""
                                ### Recommended Actions:
                                - Isolate affected plants to prevent spread
                                - Remove severely infected leaves
                                - Consider appropriate fungicides/bactericides
                                - Monitor nearby plants for similar symptoms
                                - Consult with local agricultural extension for treatment options
                                """)

# About Page
elif app_mode == "About":
    st.title("📚 About This Project")
    
    st.markdown("""
    ### Plant Disease Recognition System
    
    This application uses deep learning to identify 38 different classes of plant diseases 
    from leaf images. The goal is to help farmers and gardeners detect plant diseases early, 
    enabling timely treatment and preventing significant crop losses.
    """)
    
    st.subheader("Dataset Information")
    st.markdown("""
    The model was trained on a dataset of approximately 87,000 RGB images of healthy and 
    diseased crop leaves, categorized into 38 different classes.
    
    - **Training set:** 70,295 images
    - **Validation set:** 17,572 images
    - **Test set:** 33 images
    
    The dataset covers various plants including apples, grapes, tomatoes, corn, and more.
    """)
    
    st.subheader("Technology Stack")
    st.markdown("""
    - **Backend:** TensorFlow, Keras
    - **Frontend:** Streamlit
    - **Model:** Convolutional Neural Network (CNN)
    """)
    
    st.subheader("Disclaimer")
    st.warning("""
    This application provides preliminary disease identification only. 
    For critical agricultural decisions, always consult with certified plant pathologists 
    or agricultural experts. The developers are not responsible for any crop losses or 
    treatment outcomes based on this tool's recommendations.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Credits:**
- PlantVillage Dataset
- TensorFlow
- Streamlit
""")

# Add some custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSpinner>div {
        text-align: center;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
