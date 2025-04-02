import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Function to load model with error handling
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model at startup (cached to avoid reloading)
@st.cache_resource
def get_model():
    model_path = "trained_plant_disease_model.keras"  # Updated model filename to match your code
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return load_model(model_path)

# Tensorflow Model Prediction with improved error handling
def model_prediction(test_image):
    try:
        model = get_model()
        if model is None:
            return -1  # Error case
        
        # Load and preprocess image
        image = Image.open(test_image)
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        
        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # return index of max element
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return -1  # Error case

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM 🌱")
    
    # Try to load home image with fallback
    try:
        image_path = "home_page.jpeg"
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning("Home page image not found. Using placeholder.")
            # You can add a placeholder image here if you want
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About ℹ️")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this [GitHub repo](https://github.com/spMohanty/PlantVillage-Dataset).
    
    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    
    A new directory containing 33 test images is created later for prediction purpose.
    
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition 🔍")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            try:
                st.image(test_image, use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        
        # Predict button
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                st.snow()
                result_index = model_prediction(test_image)
                
                if result_index != -1:  # Only show results if prediction was successful
                    st.write("Our Prediction")
                    
                    # Class names (fixed formatting issues from original)
                    class_name = [
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
                    
                    if 0 <= result_index < len(class_name):
                        st.success(f"Model predicts: {class_name[result_index]}")
                    else:
                        st.error("Prediction result out of range")
