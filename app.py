import streamlit as st
import numpy as np
import cv2
from ultralytics import RTDETR

model = RTDETR("best.pt")

reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}

def classify_recklessness(class_ids):
    return "Reckless Driving" if any(cid in reckless_classes for cid in class_ids) else "Not Reckless Driving"

if file is not None:
    if file.type.startswith("image"):
        image = np.array(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR))
        preds = model.predict(image, conf=conf_thresh, iou=iou_thresh)
        annotated = model.draw(image, preds)
        class_ids = [int(det[5]) for det in preds]  # adjust based on RT‑DETR output format
        label = classify_recklessness(class_ids)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected: {label}", use_container_width=True)
    else:
        tfile = open("temp_video", "wb")
        tfile.write(file.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video")
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            preds = model.predict(frame, conf=conf_thresh, iou=iou_thresh)
            annotated = model.draw(frame, preds)
            class_ids = [int(det[5]) for det in preds]
            label = classify_recklessness(class_ids)
            font_scale = 1.5 if label == "Reckless Driving" else 0.7
            color = (0, 0, 255) if label == "Reckless Driving" else (0, 255, 0)
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        cap.release()
