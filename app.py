import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import requests

st.set_page_config(page_title="SafetyVision AI", page_icon="🪖")

st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

# Download model from GitHub if not exists
@st.cache_resource
def download_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI model... This may take a minute..."):
            # Using a direct download link (you'll need to upload best.pt to GitHub Releases)
            url = "https://github.com/Zaheenahamd78/safetyvision-helmet-detection/releases/download/v1.0/best.pt"
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error("Model download failed. Please check GitHub Release.")
                    return None
            except:
                st.error("Network error. Using fallback mode...")
                return None
    return model_path

# Load model
@st.cache_resource
def load_model():
    model_path = download_model()
    if model_path and os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning("⚠️ Using dummy model for demo")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
else:
    st.info("ℹ️ Demo mode - Model not available")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    results = model(img_array)
    annotated_img = results[0].plot()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original")
    with col2:
        st.image(annotated_img, caption="Detection")
    
    boxes = results[0].boxes
    if boxes is not None:
        helmet_count = sum(1 for box in boxes if int(box.cls[0]) == 0)
        person_count = sum(1 for box in boxes if int(box.cls[0]) == 1)
        
        st.markdown("### 📊 Detection Summary")
        st.success(f"✅ Helmets: {helmet_count}")
        st.warning(f"👷 Persons: {person_count}")
        
        if helmet_count < person_count:
            st.error("⚠️ WARNING: Workers without helmets detected!")
        else:
            st.success("🎉 All safe!")