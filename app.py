import streamlit as st
import cv2
import numpy as np
from ultralytics import RTDETR
from tempfile import NamedTemporaryFile
from pathlib import Path

st.set_page_config(page_title="Reckless Driving Behaviours", layout="wide")
st.title("🚗 Reckless Driving Behaviours — RT‑DETR")

@st.cache_resource
def load_model():
    return RTDETR("best.pt")

model = load_model()

file = st.sidebar.file_uploader(
    "Choose an image or video",
    type=["jpg", "jpeg", "png", "mp4", "mov"]
)
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---\nBy Jhillian Millare Cabos")

reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}
safe_classes = {0}

def classify_recklessness(cls_ids):
    return "Reckless Driving" if any(cid in reckless_classes for cid in cls_ids) else "Not Reckless Driving"

if file is None:
    st.info("⬅️ Upload an image or video to begin.")
    st.stop()

with NamedTemporaryFile(delete=False) as tmp:
    tmp.write(file.read())
    temp_path = Path(tmp.name)

if file.type.startswith("image"):
    im = cv2.imread(str(temp_path))
    results = model(im, conf=confidence, iou=iou_thresh, verbose=False)[0]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    verdict = classify_recklessness(cls_ids)
    st.subheader(verdict)
    plotted = results.plot()
    st.image(plotted, channels="BGR", caption=verdict)

elif file.type.startswith("video"):
    cap = cv2.VideoCapture(str(temp_path))
    stframe = st.empty()
    verdict_overall = "Not Reckless Driving"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=confidence, iou=iou_thresh, verbose=False)[0]
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        frame_verdict = classify_recklessness(cls_ids)
        if frame_verdict == "Reckless Driving":
            verdict_overall = "Reckless Driving"
        annotated = results.plot()
        stframe.image(annotated, channels="BGR")

    cap.release()
    st.subheader(f"Overall verdict: {verdict_overall}")

else:
    st.error("Unsupported file type.")
