import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import tempfile
import os

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Detect, track, count, and save objects in real - time.")

enable_count = st.checkbox("Show object count", value = True)
enable_save = st.checkbox("Save Frame", value = False)
detect_person = st.checkbox("Alert if Person is Detected", value = False)

temp_dir = tempfile.gettempdir()

def video_frame_callback(frame):
    img = frame.to_ndarray(format = "bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    if enable_count:
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(
            annotated_frame,
            f"objects: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    if detect_person and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                cv2.putText(
                 annotated_frame,
                 "PERSON DETECTED!",
                 (10, 70),
                 cv2.FONT_HERSHEY_SIMPLEX,
                  1,
                  (0, 0, 255),
                  2,
        )

    if enable_save:
        file_path = os.path.join(temp_dir, "detected_frame.jpg")
        cv2.imwrite(file_path, annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,  # <-- REQUIRED
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)