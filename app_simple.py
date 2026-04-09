import streamlit as st
import os

st.title("SafetyVision AI")
st.write("Checking if model exists...")

if os.path.exists("best.pt"):
    st.success("✅ best.pt found!")
    st.write(f"File size: {os.path.getsize('best.pt') / 1024 / 1024:.2f} MB")
else:
    st.error("❌ best.pt not found!")
    st.write("Current files:")
    for file in os.listdir("."):
        st.write(f"- {file}")