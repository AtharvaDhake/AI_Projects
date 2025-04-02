import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
MODEL_PATH = 'my_model.keras'
INPUT_SIZE = (128, 128)  # Update if your model expects different dimensions

# Complete list of all 38 classes in correct order
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
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed. Common issues:
        1. Model file not found at: {MODEL_PATH}
        2. TensorFlow version mismatch
        3. Corrupted model file
        Error: {str(e)}
        """)
        return None

def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert('RGB')
        img = img.resize(INPUT_SIZE)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Diagnose", "Class List"])

    if app_mode == "Home":
        st.header("🌿 Plant Disease Classification System")
        st.markdown("""
        ### How to Use:
        1. Go to **Diagnose** page
        2. Upload a clear image of a plant leaf
        3. Get instant disease classification
        """)
        st.write("")  # Spacer
        st.image("https://via.placeholder.com/800x400?text=Plant+Disease+Classifier", 
                use_column_width=True)

    elif app_mode == "Class List":
        st.header("📋 Complete Class List (38 Categories)")
        st.write("This model can identify the following conditions:")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {class_name.replace('___', ' - ').replace('_', ' ')}")

    elif app_mode == "Diagnose":
        st.header("🔍 Disease Diagnosis")
        model = load_model()
        
        uploaded_file = st.file_uploader(
            "Choose a leaf image", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze") and model:
                with st.spinner("Analyzing..."):
                    try:
                        # Preprocess and predict
                        processed_img = preprocess_image(uploaded_file)
                        if processed_img is not None:
                            predictions = model.predict(processed_img)
                            pred_index = np.argmax(predictions)
                            confidence = np.max(predictions)
                            
                            # Display results
                            with col2:
                                st.subheader("Diagnosis Results")
                                st.write(f"**Condition:** {CLASS_NAMES[pred_index].replace('___', ' - ').replace('_', ' ')}")
                                st.write(f"**Confidence:** {confidence:.1%}")
                                
                                if "healthy" in CLASS_NAMES[pred_index]:
                                    st.success("This plant appears healthy!")
                                    st.balloons()
                                else:
                                    st.error("Potential disease detected!")
                                    st.warning("Recommendation: Consult an agricultural expert")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
