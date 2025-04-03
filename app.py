import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set up the app
st.set_page_config(page_title="Plant Doctor 🌱", layout="centered")

# Load your model (make sure it's in the same folder)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('my_model.h5')
        return model
    except:
        st.error("❌ Couldn't load the model. Make sure 'trained_model.h5' is in the same folder.")
        return None

# Your class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Simple image preprocessing
def prepare_image(image):
    try:
        # Open and resize image
        img = Image.open(image).convert('RGB')
        img = img.resize((128, 128))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img, img_array
    except:
        st.error("❌ Couldn't process the image")
        return None, None

# Main app
def main():
    st.title("🌱 Plant Disease Classifier")
    st.write("Upload a photo of a plant leaf to check for diseases")
    
    # Load model
    model = load_model()
    if not model:
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the image
        st.image(uploaded_file, caption="Your Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze"):
            with st.spinner("Checking for diseases..."):
                # Prepare the image
                original_img, processed_img = prepare_image(uploaded_file)
                
                if processed_img is not None:
                    try:
                        # Make prediction
                        prediction = model.predict(processed_img)
                        result_index = np.argmax(prediction)
                        disease = CLASS_NAMES[result_index]
                        confidence = np.max(prediction) * 100
                        
                        # Show results
                        st.success(f"**Diagnosis:** {disease.replace('___', ' - ')}")
                        st.info(f"**Confidence:** {confidence:.1f}%")
                        
                        # Special messages
                        if "healthy" in disease.lower():
                            st.balloons()
                            st.success("🎉 Your plant looks healthy!")
                        else:
                            st.warning("⚠️ Disease detected! Consider consulting a plant specialist.")
                            
                    except:
                        st.error("❌ Failed to make prediction")

if __name__ == "__main__":
    main()
