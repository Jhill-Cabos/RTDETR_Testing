import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import RTDETR

model = RTDETR("best.pt")

st.set_page_config(page_title="Reckless Driving Behaviours using YOLOV11", layout="wide")
st.title("🚗 Reckless Driving Behaviours using YOLOV11")

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

    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        results = model(image_np, conf=confidence, iou=iou_thresh)
        annotated_img = np.squeeze(results[0].plot())

        class_ids = [int(result.cls) for result in results[0].boxes]
        label = classify_recklessness(class_ids)
        st.text(f"Detected: {label}")

        st.image(annotated_img, caption="Detection Result with Classification", use_container_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated_frame = np.squeeze(results[0].plot())

            class_ids = [int(result.cls) for result in results[0].boxes]
            label = classify_recklessness(class_ids)

            for result in results[0].boxes:
                xmin, ymin, xmax, ymax = map(int, result.xyxy[0].cpu().numpy())
                font_scale = 1.5 if label == "Reckless Driving" else 0.7
                color = (0, 0, 255) if label == "Reckless Driving" else (0, 255, 0)
                label_position = (xmax - 150, ymax + 30)
                cv2.putText(annotated_frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
