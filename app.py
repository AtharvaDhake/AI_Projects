import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
MODEL_PATH = 'my_model.keras'
INPUT_SIZE = (128, 128)

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
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize(INPUT_SIZE)
    return np.array(img) / 255.0

def main():
    st.set_page_config(page_title="Plant Disease Detector")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Diagnosis"])

    if app_mode == "Home":
        st.header("🌱 Plant Disease Recognition System")
        st.image("home_page.jpeg", use_column_width=True)
        st.markdown("""
        ### How to Use:
        1. Upload a clear leaf image in Diagnosis page
        2. Get instant disease detection
        """)

    elif app_mode == "About":
        st.header("About")
        st.markdown(f"""
        - **Classes**: {len(CLASS_NAMES)} plant conditions
        - **Input Size**: {INPUT_SIZE[0]}x{INPUT_SIZE[1]} pixels
        """)

    elif app_mode == "Diagnosis":
        st.header("Diagnosis")
        model = load_model()
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file and model:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner("Processing..."):
                    try:
                        img_array = preprocess_image(uploaded_file)
                        input_arr = np.expand_dims(img_array, axis=0)
                        predictions = model.predict(input_arr)
                        result_idx = np.argmax(predictions)
                        
                        with col2:
                            st.success(f"**Result**: {CLASS_NAMES[result_idx]}")
                            if "healthy" in CLASS_NAMES[result_idx]:
                                st.balloons()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
