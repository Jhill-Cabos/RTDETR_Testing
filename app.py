import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import RTDETR

# Load RT-DETR model
model = RTDETR("best.pt")

# Set correct class names
model.names = {
    0: 'c0 - Safe Driving',
    1: 'c1 - Texting',
    2: 'c2 - Talking on the phone',
    3: 'c3 - Operating the Radio',
    4: 'c4 - Drinking',
    5: 'c5 - Reaching Behind',
    6: 'c6 - Hair and Makeup',
    7: 'c7 - Talking to Passenger',
    8: 'd0 - Eyes Closed',
    9: 'd1 - Yawning',
    10: 'd2 - Nodding Off',
    11: 'd3 - Eyes Open',
    12: 'e1 - Seat Belt'
}

# Streamlit UI setup
st.set_page_config(page_title="RT-DETR Driving Behavior Detection", layout="wide")
st.title("🚗 Dangerous Driving Behavior Detection (RT-DETR)")

file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold:", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit and RT-DETR")

if file is not None:
    file_type = file.type

    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert PIL image to BGR numpy array
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model(image_bgr, conf=confidence, iou=iou_thresh)
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detection Result", use_container_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        out_path = os.path.join("outputs", os.path.basename(video_path))
        os.makedirs("outputs", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        out.release()
        st.success("Video processing complete!")

        st.video(out_path)
