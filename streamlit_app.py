import streamlit as st 
import numpy as np 
from PIL import Image 
import os 
 
st.set_page_config(page_title="SafetyVision AI", page_icon="??") 
 
st.title("?? Real-time Construction Site Safety Monitor") 
st.markdown("Upload an image to detect helmets and people") 
 
def load_model(): 
    from ultralytics import YOLO 
    model_path = "best.pt" 
    if os.path.exists(model_path): 
        return YOLO(model_path) 
    else: 
        st.error("Model file not found!") 
        return None 
model = load_model() 
if model: 
    st.success("? Model loaded successfully!") 
 
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 
 
if uploaded_file is not None and model: 
    image = Image.open(uploaded_file) 
    img_array = np.array(image) 
    results = model(img_array) 
    annotated_img = results[0].plot() 
    col1, col2 = st.columns(2) 
    with col1: 
        st.image(image, caption="Original Image") 
    with col2: 
        st.image(annotated_img, caption="Detection Result") 
    boxes = results[0].boxes 
    boxes = results[0].boxes 
    if boxes is not None and len(boxes) 
        st.info("No objects detected") 
