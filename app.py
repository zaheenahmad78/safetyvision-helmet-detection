import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="Helmet Safety Detection", layout="wide")
st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at: {model_path}")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Run detection
        with st.spinner("🔍 Detecting helmets..."):
            results = model(tmp_path)
        
        # Display results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
        
        # Get the annotated image
        annotated_img = results[0].plot()
        
        with col2:
            st.image(annotated_img, caption="Detection Result", use_container_width=True)
        
        # Show statistics
        if results[0].boxes is not None:
            boxes = results[0].boxes
            class_names = results[0].names
            
            # Count persons and helmets
            person_count = 0
            helmet_count = 0
            
            for box in boxes:
                cls_id = int(box.cls[0])
                if class_names[cls_id].lower() == 'person':
                    person_count += 1
                elif 'helmet' in class_names[cls_id].lower():
                    helmet_count += 1
            
            st.success(f"📊 **Results:** {person_count} persons detected | {helmet_count} helmets detected")
        
        # Cleanup temp file
        os.unlink(tmp_path)
else:
    st.error("Failed to load model. Please check your model file.")