import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
from ultralytics.utils import LOGGER
LOGGER.setLevel('ERROR')

st.title("🎥 Real-Time Object Detection (YOLOv8)")

# model = YOLO("yolov8n.pt")  # lightweight model
model = YOLO("yolov8m.pt")

video_source = st.radio("Choose video source", ["Webcam", "Upload Video"])

if video_source == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break
        results = model(frame, imgsz=1280, conf=0.6)
        annotated = results[0].plot()
        FRAME_WINDOW.image(annotated, channels="BGR")
    cap.release()

else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=640, conf=0.4)
            annotated = results[0].plot()
            FRAME_WINDOW.image(annotated, channels="BGR")
        cap.release()