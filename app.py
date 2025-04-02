import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants - Update these paths as needed
MODEL_PATH = 'my_model.keras'  # Changed from .h5 to .keras
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

# Load model with .keras format
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Please ensure:")
        st.error("1. The model file (my_model.keras) exists in the correct location")
        st.error("2. You have TensorFlow 2.10 or higher installed")
        return None

def preprocess_image(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((128, 128))  # Ensure this matches your model's expected input
    return np.array(image) / 255.0  # Normalize to [0,1]

def predict_disease(model, image):
    input_arr = np.expand_dims(image, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)

def main():
    st.set_page_config(page_title="Plant Disease Detector", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "About", "Diagnosis"])

    if app_mode == "Home":
        st.header("🌱 Plant Disease Recognition System")
        st.image("home_page.jpg", use_column_width=True)
        st.markdown("""
        ### How to Use:
        1. Go to **Diagnosis** page
        2. Upload a clear image of a plant leaf
        3. Get instant disease detection
        """)

    elif app_mode == "About":
        st.header("About This Project")
        st.markdown("""
        - **Model**: CNN trained on 87K+ plant images
        - **Classes**: 38 different plant disease categories
        - **Format**: Keras SavedModel (.keras)
        """)

    elif app_mode == "Diagnosis":
        st.header("Plant Disease Diagnosis")
        model = load_model()  # Load model at start of diagnosis page
        
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze") and model:  # Only enable if model loaded
                with st.spinner("Analyzing..."):
                    try:
                        image = preprocess_image(uploaded_file)
                        result_idx, confidence = predict_disease(model, image)
                        
                        with col2:
                            st.success(f"**Prediction**: {CLASS_NAMES[result_idx]}")
                            st.progress(float(confidence))
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                            
                            if "healthy" in CLASS_NAMES[result_idx]:
                                st.balloons()
                            else:
                                st.warning("This plant shows disease symptoms!")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
