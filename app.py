import streamlit as st
import numpy as np
import cv2
from ultralytics import RTDETR

model = RTDETR("best.pt")

st.set_page_config(page_title="Reckless Driving Behaviours", layout="wide")
st.title("🚗 Reckless Driving Behaviours RTDETR")

file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold:", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("By Jhillian Millare Cabos")

reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}
safe_classes = {0}

def classify_recklessness(class_ids):
    if any(class_id in reckless_classes for class_id in class_ids):
        return "Reckless Driving"
    else:
        return "Not Reckless Driving"

if file is not None:
    file_type = file.type
