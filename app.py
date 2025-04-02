import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = 'my_model.keras'
INPUT_SIZE = (128, 128)  # Update this to match your model's expected input size

# Properly ordered class names
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
        model.summary()  # This will print in your terminal where Streamlit runs
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("Please check:")
        st.error("1. Model file exists at the specified path")
        st.error("2. TensorFlow version matches the model's training version")
        st.error("3. File isn't corrupted")
        return None

def preprocess_image(image):
    try:
        img = Image.open(image).convert("RGB")
        st.write("Original image size:", img.size)  # Debug original dimensions
        
        # Resize to model's expected input
        img = img.resize(INPUT_SIZE)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        
        # Debug display
        fig, ax = plt.subplots()
        ax.imshow(img_array)
        ax.set_title("Preprocessed Image")
        st.pyplot(fig)
        
        return img_array
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

def predict_disease(model, image):
    try:
        input_arr = np.expand_dims(image, axis=0)
        st.write("Input array shape:", input_arr.shape)  # Debug shape
        
        predictions = model.predict(input_arr)
        st.write("Raw predictions:", predictions)  # Debug raw output
        
        result_idx = np.argmax(predictions)
        return result_idx, predictions
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="Plant Disease Detector", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Diagnosis", "Debug"])

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
        st.markdown(f"""
        - **Model**: CNN trained on plant images
        - **Classes**: {len(CLASS_NAMES)} different categories
        - **Input Size**: {INPUT_SIZE[0]}x{INPUT_SIZE[1]} pixels
        """)

    elif app_mode == "Diagnosis":
        st.header("Plant Disease Diagnosis")
        model = load_model()
        
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file and model:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    try:
                        image = preprocess_image(uploaded_file)
                        if image is not None:
                            result_idx, predictions = predict_disease(model, image)
                            
                            if result_idx is not None:
                                with col2:
                                    st.success(f"**Prediction**: {CLASS_NAMES[result_idx]}")
                                    
                                    if "healthy" in CLASS_NAMES[result_idx]:
                                        st.balloons()
                                    else:
                                        st.warning("Disease detected!")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    elif app_mode == "Debug":
        st.header("🛠️ Debug Mode")
        st.warning("This page shows technical details for troubleshooting")
        
        model = load_model()
        if model:
            st.success("✅ Model loaded successfully")
            st.subheader("Model Architecture")
            st.text(model.summary())
            
            st.subheader("Sample Test")
            test_image = np.random.rand(*INPUT_SIZE, 3)  # Random test image
            st.write("Test image shape:", test_image.shape)
            
            try:
                prediction = model.predict(np.expand_dims(test_image, axis=0))
                st.write("Test prediction output:", prediction)
            except Exception as e:
                st.error(f"Test prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
