import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# Load model with caching and verification
@st.cache_resource
def load_model():
    model_path = "my_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        # Verify model can make predictions
        dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        _ = model.predict(dummy_input)
        return model
    except Exception as e:
        st.error(f"Error loading or verifying model: {str(e)}")
        return None

# Class names (ensure these exactly match your training labels)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def preprocess_image(image):
    """Match exactly how you preprocessed images during training"""
    try:
        # Open and convert to RGB (important for PNGs with alpha channel)
        img = Image.open(image).convert('RGB')
        
        # Resize using the exact same method as training
        img = img.resize((128, 128), Image.BILINEAR)  # or whatever you used
        
        # Convert to array and normalize EXACTLY like training
        img_array = np.array(img) / 255.0  # if you used 0-1 normalization
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def predict_disease(image, model):
    try:
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, None
            
        predictions = model.predict(processed_img)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Debug function to compare with your test script
def debug_prediction(image_path, model):
    """Use this to verify your Streamlit preprocessing matches your test script"""
    try:
        # Load image the way your test script does
        test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        test_array = tf.keras.preprocessing.image.img_to_array(test_img)
        test_array = np.expand_dims(test_array, axis=0) / 255.0
        
        # Get test script prediction
        test_pred = model.predict(test_array)
        test_class = CLASS_NAMES[np.argmax(test_pred)]
        
        # Get Streamlit prediction
        with open(image_path, 'rb') as f:
            streamlit_class, _ = predict_disease(f, model)
        
        return {
            'test_script_prediction': test_class,
            'streamlit_prediction': streamlit_class,
            'match': test_class == streamlit_class
        }
    except Exception as e:
        return {'error': str(e)}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "Debug"])

# Main detection page
if page == "Detection":
    st.title("🔍 Plant Disease Detection")
    model = load_model()
    
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict Disease"):
            with st.spinner("Analyzing..."):
                disease, confidence = predict_disease(uploaded_file, model)
                
                if disease:
                    with col2:
                        st.subheader("Results")
                        st.success(f"**Prediction:** {disease}")
                        st.info(f"**Confidence:** {confidence:.2f}%")
                        
                        # Show raw prediction values for debugging
                        with st.expander("Debug Info"):
                            processed_img = preprocess_image(uploaded_file)
                            predictions = model.predict(processed_img)
                            st.write("Raw prediction values:", predictions)
                            st.write("Class indices:", np.argsort(predictions[0])[::-1])

# Debug page to compare with test script
elif page == "Debug" and load_model() is not None:
    st.title("🐛 Debug Predictions")
    st.warning("Compare Streamlit preprocessing with your test script")
    
    test_image_path = st.text_input("Path to test image (from your test script):")
    if test_image_path and os.path.exists(test_image_path):
        debug_info = debug_prediction(test_image_path, load_model())
        st.json(debug_info)
