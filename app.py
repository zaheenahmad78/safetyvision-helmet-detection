import streamlit as st
import numpy as np
from PIL import Image
import os
import torch

st.set_page_config(page_title="SafetyVision AI", page_icon="🪖")

st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        
        # Try to load custom model
        model_path = "best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            return model
        else:
            st.warning("Custom model not found. Using default YOLOv8n model for demo.")
            return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(annotated_img, caption="Detection Result")
        
        # Process results
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            helmet_count = 0
            person_count = 0
            
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    helmet_count += 1
                elif cls_id == 1:
                    person_count += 1
            
            st.markdown("### 📊 Detection Summary")
            st.success(f"✅ Helmets: {helmet_count}")
            st.warning(f"👷 Persons: {person_count}")
            
            if helmet_count < person_count and person_count > 0:
                st.error("⚠️ WARNING: Workers without helmets detected!")
            elif helmet_count >= person_count and person_count > 0:
                st.success("🎉 All workers safe!")
            else:
                st.info("No persons detected")
        else:
            st.info("No objects detected in this image")
            
    except Exception as e:
        st.error(f"Error during detection: {e}")
        st.info("Try uploading a different image (JPG or PNG format)")
