import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide",
    page_icon="🌱"
)

# Constants (copied from your test code)
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

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('trained_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# Image preprocessing (exactly matching your test code)
def preprocess_image(image_path):
    try:
        # Method 1: Using cv2 like in your test code
        img_cv = cv2.imread(image_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Method 2: Using tf.keras.preprocessing like in your test code
        img_tf = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img_tf)
        input_arr = np.array([input_arr])  # Convert single image to a batch
        
        return img_cv, input_arr
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return None, None

# Main app function
def main():
    st.title("🌱 Plant Disease Classifier")
    st.markdown("Upload an image of a plant leaf to detect potential diseases")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = "temp_uploaded_image.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process image exactly like your test code
        original_img, processed_img = preprocess_image(temp_file_path)
        
        if original_img is not None and processed_img is not None:
            # Display original image
            st.subheader("Uploaded Image")
            st.image(original_img, caption="Original Image", use_column_width=True)
            
            # Make prediction
            if st.button("🔍 Analyze Image"):
                with st.spinner("Processing..."):
                    try:
                        # Predict (exactly like your test code)
                        prediction = model.predict(processed_img)
                        result_index = np.argmax(prediction)
                        model_prediction = CLASS_NAMES[result_index]
                        confidence = np.max(prediction) * 100
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_img, 
                                    caption=f"Prediction: {model_prediction}",
                                    use_column_width=True)
                        
                        with col2:
                            st.success(f"**Predicted Disease:** {model_prediction}")
                            st.info(f"**Confidence:** {confidence:.2f}%")
                            
                            # Show raw prediction values for debugging
                            with st.expander("Advanced Details"):
                                st.write("Prediction array shape:", prediction.shape)
                                st.write("Raw prediction values:", prediction)
                                st.write("Top 3 predictions:")
                                top3_indices = np.argsort(prediction[0])[-3:][::-1]
                                for idx in top3_indices:
                                    st.write(f"- {CLASS_NAMES[idx]}: {prediction[0][idx]*100:.2f}%")
                        
                        # Add some helpful information based on prediction
                        if "healthy" in model_prediction.lower():
                            st.balloons()
                            st.success("🎉 The plant appears healthy! No disease detected.")
                        else:
                            st.warning("⚠️ Potential disease detected. Consider consulting with a plant pathologist.")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                
                # Clean up temporary file
                os.remove(temp_file_path)

# Run the app
if __name__ == "__main__":
    main()
