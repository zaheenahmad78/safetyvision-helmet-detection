import streamlit as st
import os
import tempfile

st.set_page_config(page_title="Helmet Safety Detection", layout="wide")
st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

# Set environment variables before importing ultralytics
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import ultralytics
from ultralytics import YOLO

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("Detecting..."):
            results = model(tmp_path)
        
        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Original", use_container_width=True)
        col2.image(results[0].plot(), caption="Detection", use_container_width=True)
        
        if results[0].boxes:
            boxes = results[0].boxes
            class_names = results[0].names
            person_count = sum(1 for c in boxes.cls if class_names[int(c)].lower() == 'person')
            helmet_count = sum(1 for c in boxes.cls if 'helmet' in class_names[int(c)].lower())
            st.success(f"👷 Persons: {person_count} | 🪖 Helmets: {helmet_count}")
        
        os.unlink(tmp_path)
else:
    st.error("Failed to load model")