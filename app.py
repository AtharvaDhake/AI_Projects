import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Simple title
st.title("🌱 Plant Disease Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prediction button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing..."):
            try:
                # Load model (replace with your actual model loading code)
                model = tf.keras.models.load_model("my_model.keras")
                
                # Preprocess image
                img = image.resize((128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)
                
                # Simple class names (modify with your actual classes)
                class_names = [
                    "Healthy", 
                    "Apple Scab", 
                    "Tomato Blight",
                    "Leaf Spot"
                ]
                
                # Show result
                st.success(f"Prediction: {class_names[predicted_class]}")
                st.write(f"Confidence: {np.max(prediction)*100:.2f}%")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Make sure 'my_model.keras' exists in your directory")

# Minimal instructions
st.markdown("""
### How to use:
1. Upload a clear photo of a plant leaf
2. Click "Predict Disease"
3. View the results
""")
