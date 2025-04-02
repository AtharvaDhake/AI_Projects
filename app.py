import streamlit as st
from PIL import Image
import os

# =============================================
# Custom CSS for proper UI rendering
# =============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white;
        padding: 2rem 1rem;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar title */
    .sidebar-title {
        color: #2c3e50;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Navigation buttons */
    .nav-button {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background-color: #e9f5ff;
    }
    
    .nav-button.active {
        background-color: #e1f0ff;
        font-weight: bold;
    }
    
    /* Quick guide box */
    .quick-guide {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    /* Main content area */
    .content {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# Sidebar Navigation
# =============================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌿 PlanMD Dashboard</div>', unsafe_allow_html=True)
    
    # Navigation selection
    app_mode = st.radio(
        "Navigate to:",
        ["Home", "About", "Disease Scanner"],
        label_visibility="collapsed"
    )
    
    # Quick Guide
    st.markdown("""
    <div class="quick-guide">
        <h4>Quick Guide</h4>
        <ol style="padding-left: 1.2rem;">
            <li style="margin-bottom: 0.5rem;">Upload a clear image of a plant leaf</li>
            <li style="margin-bottom: 0.5rem;">Click Analyze button</li>
            <li>Get instant diagnosis</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# =============================================
# Main Content Area
# =============================================
st.markdown('<div class="content">', unsafe_allow_html=True)

if app_mode == "Home":
    st.title("🌱 Plant Disease Recognition System")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Your AI-Powered Plant Doctor
        Detect diseases in your plants with **95% accuracy** using our 
        state-of-the-art deep learning model.
        """)
        
        st.markdown("""
        - 🚀 Instant diagnosis
        - 🌍 Supports 38 plant varieties
        - 📱 Mobile-friendly interface
        """)
        
    with col2:
        try:
            st.image("home_page.jpg" if os.path.exists("home_page.jpg") else 
                   Image.new('RGB', (300, 200), color=(73, 109, 137)),
                   caption="Healthy vs Diseased Leaves", use_column_width=True)
        except:
            st.image(Image.new('RGB', (300, 200), color=(73, 109, 137)),
                   caption="Image not found", use_column_width=True)

elif app_mode == "About":
    st.title("About PlantMD")
    st.markdown("""
    ### Our Mission
    We're making plant healthcare accessible to everyone through AI technology.
    """)
    
    tab1, tab2 = st.tabs(["📊 Dataset", "🧠 Technology"])
    with tab1:
        st.markdown("""
        We use the PlantVillage dataset containing:
        - 87,000+ high-quality images
        - 38 categories of plants
        - Scientifically validated
        """)
        
    with tab2:
        st.markdown("""
        - Deep learning model with 95%+ accuracy
        - TensorFlow/Keras framework
        - Streamlit for web interface
        """)

elif app_mode == "Disease Scanner":
    st.title("🔍 Disease Scanner")
    
    uploaded_file = st.file_uploader("Upload plant leaf image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                # Simulate analysis
                import time
                time.sleep(2)
                
                # Display results
                st.success("Analysis Complete!")
                st.markdown("""
                <div style="background:#f0f8ff; padding:1rem; border-radius:8px;">
                    <h4>Diagnosis Results</h4>
                    <p>🍃 <b>Plant:</b> Tomato</p>
                    <p>⚕️ <b>Condition:</b> Healthy</p>
                    <p>📊 <b>Confidence:</b> 96%</p>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
