import streamlit as st
import insightface
import cv2
import numpy as np

st.title("Test InsightFace and OpenCV Import")

try:
    st.write("Attempting to initialize InsightFace...")
    face_analysis = insightface.app.FaceAnalysis(det_name='retinaface_r50_v1', rec_name='arcface_r100_v1')
    face_analysis.prepare(ctx_id=-1, det_size=(640, 640))
    st.success("InsightFace initialized successfully!")
except Exception as e:
    st.error(f"InsightFace initialization failed: {str(e)}")

st.write(f"OpenCV version: {cv2.__version__}")
st.write(f"NumPy version: {np.__version__}")