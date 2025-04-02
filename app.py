import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Constants - MUST VERIFY THESE MATCH YOUR MODEL
MODEL_PATH = 'my_model.keras'
INPUT_SIZE = (128, 128)  # Update to your model's expected input size
CHANNELS = 3  # 3 for RGB, 1 for grayscale

# Class names - MUST match your model's training order EXACTLY
CLASS_NAMES = [
   'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Test prediction with dummy data
        test_input = np.random.rand(1, *INPUT_SIZE, CHANNELS)
        test_pred = model.predict(test_input)
        st.success("✅ Model loaded and test prediction successful!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def preprocess_image(image):
    try:
        img = Image.open(image).convert("RGB" if CHANNELS == 3 else "L")
        
        # Debug original image
        st.write("Original image mode:", img.mode, "size:", img.size)
        
        # Resize and normalize
        img = img.resize(INPUT_SIZE)
        img_array = np.array(img) / 255.0
        
        # Verify preprocessing
        st.write("Processed image shape:", img_array.shape)
        st.write("Pixel range:", np.min(img_array), "to", np.max(img_array))
        
        return img_array
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

def predict_with_verification(model, image_array):
    try:
        # Verify input dimensions
        if image_array.ndim == 3:
            input_arr = np.expand_dims(image_array, axis=0)
        elif image_array.ndim == 4:
            input_arr = image_array
        else:
            raise ValueError(f"Unexpected image dimensions: {image_array.ndim}")
            
        st.write("Model input shape:", input_arr.shape)
        
        # Get predictions
        predictions = model.predict(input_arr)
        st.write("Raw predictions:", predictions)
        
        # Get top 3 predictions
        top_k = 3
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        return top_indices, predictions
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="Plant Disease Detector", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "Diagnosis", "Debug"])

    if app_mode == "Home":
        st.header("🌱 Plant Disease Recognition System")
        st.image("home_page.jpg", use_column_width=True)

    elif app_mode == "Diagnosis":
        st.header("Plant Disease Diagnosis")
        model = load_model()
        
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file and model:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner("Processing..."):
                    img_array = preprocess_image(uploaded_file)
                    
                    if img_array is not None:
                        top_indices, predictions = predict_with_verification(model, img_array)
                        
                        if top_indices is not None:
                            with col2:
                                st.subheader("Top Predictions:")
                                for i, idx in enumerate(top_indices):
                                    st.write(f"{i+1}. {CLASS_NAMES[idx]} (score: {predictions[0][idx]:.4f})")
                                
                                # Visual feedback
                                best_pred = CLASS_NAMES[top_indices[0]]
                                if "healthy" in best_pred:
                                    st.success("✅ Healthy plant detected!")
                                    st.balloons()
                                else:
                                    st.error(f"⚠️ Potential disease: {best_pred}")

    elif app_mode == "Debug":
        st.header("🛠️ Debug Console")
        model = load_model()
        
        if model:
            st.subheader("Model Summary")
            # Create a text box with model summary
            summary = []
            model.summary(print_fn=lambda x: summary.append(x))
            st.text("\n".join(summary))
            
            st.subheader("Input Test")
            test_image = np.random.rand(*INPUT_SIZE, CHANNELS)
            st.write("Test image shape:", test_image.shape)
            predictions = model.predict(np.expand_dims(test_image, axis=0))
            st.write("Test predictions:", predictions)

if __name__ == "__main__":
    main()
