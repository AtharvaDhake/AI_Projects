import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stFileUploader>div>div>div>button {
        background-color: #4285F4;
        color: white;
    }
    .stMarkdown h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMarkdown h2 {
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .healthy {
        border-left: 5px solid #2ecc71;
    }
    .diseased {
        border-left: 5px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Function to load model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("my_model.keras")
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        return None

# Model prediction function with progress animation
def model_prediction(test_image):
    model = load_model()
    if model is None:
        return -1
    
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate loading steps
        for percent_complete in range(0, 101, 20):
            status_text.text(f"Analyzing... {percent_complete}%")
            progress_bar.progress(percent_complete)
            time.sleep(0.1)
        
        # Actual prediction
        image = Image.open(test_image).resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        predictions = model.predict(input_arr)
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return np.argmax(predictions)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return -1

# Sidebar with enhanced styling
with st.sidebar:
    st.title("🌿 PlantMD Dashboard")
    app_mode = st.radio(
        "Navigate to:",
        ["🏠 Home", "ℹ️ About", "🔍 Disease Scanner"],
        index=0
    )
    st.markdown("---")
    st.markdown("""
    ### Quick Guide
    1. Upload a clear image of a plant leaf
    2. Click 'Analyze' button
    3. Get instant diagnosis
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Home Page
if "🏠 Home" in app_mode:
    st.title("🌱 Plant Disease Recognition System")
    
    # Hero section with columns
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Your AI-Powered Plant Doctor
        Detect diseases in your plants with **95% accuracy** using our 
        state-of-the-art deep learning model. Save your crops before 
        it's too late!
        """)
        st.markdown("""
        - 🚀 Instant diagnosis
        - 🌍 Supports 38 plant varieties
        - 📱 Mobile-friendly interface
        """)
        
    with col2:
        try:
            st.image("home_page.jpeg" if os.path.exists("home_page.jpeg") else 
                   Image.new('RGB', (300, 200), color=(73, 109, 137)),
                   caption="Healthy vs Diseased Leaves", use_column_width=True)
        except:
            st.image(Image.new('RGB', (300, 200), color=(73, 109, 137)),
                   caption="Image not found", use_column_width=True)
    
    # Features section
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    features = st.columns(3)
    features[0].markdown("""
    ### 🔍 Accurate Detection
    Our AI model has been trained on thousands of images to provide reliable diagnoses.
    """)
    
    features[1].markdown("""
    ### ⚡ Fast Results
    Get diagnosis in seconds, not days. No waiting for lab tests!
    """)
    
    features[2].markdown("""
    ### 🌱 Plant Care Tips
    Receive customized advice for treating detected diseases.
    """)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <h3>Ready to diagnose your plants?</h3>
        <p>Click on <b>Disease Scanner</b> in the sidebar to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# About Page
elif "ℹ️ About" in app_mode:
    st.title("About PlantMD")
    
    # Animated tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🧠 Technology", "👨‍🌾 Our Mission"])
    
    with tab1:
        st.markdown("""
        ### PlantVillage Dataset
        We use the renowned PlantVillage dataset containing:
        - 87,000+ high-quality images
        - 38 categories of healthy/diseased plants
        - 14 crop species with 26 diseases
        
        The dataset is scientifically validated and widely used in agricultural research.
        """)
        st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-019-0321-1/MediaObjects/41597_2019_321_Fig1_HTML.png",
                caption="Sample images from PlantVillage dataset", use_column_width=True)
    
    with tab2:
        st.markdown("""
        ### Under the Hood
        - **Deep Learning Model**: Custom CNN architecture with 95%+ accuracy
        - **Framework**: TensorFlow/Keras
        - **Preprocessing**: Advanced image augmentation techniques
        - **Deployment**: Streamlit cloud platform
        """)
        st.code("""
        # Model Architecture
        model = Sequential([
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(38, activation='softmax')
        ])
        """)
    
    with tab3:
        st.markdown("""
        ### Our Vision
        We believe in democratizing plant healthcare through technology.
        By making disease detection accessible to everyone, we aim to:
        - Reduce crop losses worldwide
        - Minimize pesticide overuse
        - Empower small farmers
        - Promote sustainable agriculture
        """)
        st.success("Join us in making agriculture more resilient and productive!")

# Disease Detection Page
elif "🔍 Disease Scanner" in app_mode:
    st.title("Plant Disease Scanner")
    
    # Upload section with card
    with st.expander("📤 Upload Plant Image", expanded=True):
        test_image = st.file_uploader(
            "Choose an image of a plant leaf",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if test_image:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("Scanning for diseases..."):
                    # Animated progress
                    result_index = model_prediction(test_image)
                    
                    # Class names
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
                        # Result card with animation
                        result = class_name[result_index]
                        is_healthy = "healthy" in result.lower()
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="prediction-card {'healthy' if is_healthy else 'diseased'}">
                                <h2>Diagnosis Results</h2>
                                <h3>{result}</h3>
                                {"🎉 Your plant appears healthy!" if is_healthy else "⚠️ Disease detected!"}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if is_healthy:
                                st.balloons()
                                st.success("Great news! No signs of disease detected.")
                            else:
                                st.warning("Disease detected! Consider these steps:")
                                st.markdown("""
                                - Isolate the affected plant
                                - Remove severely infected leaves
                                - Apply appropriate treatment
                                - Monitor plant health regularly
                                """)
                                
                            # Show treatment tips for specific diseases
                            if "blight" in result.lower():
                                st.info("💡 Treatment Tip: Use copper-based fungicides for blight control")
                            elif "mildew" in result.lower():
                                st.info("💡 Treatment Tip: Improve air circulation and reduce leaf wetness")
                    else:
                        st.error("Unable to determine plant condition")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>PlantMD • v2.0 • Made with ❤️ for farmers everywhere</p>
    <p>Need help? contact@plantmd.example.com</p>
</div>
""", unsafe_allow_html=True)
