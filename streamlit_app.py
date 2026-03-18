import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# 1. Configuration & Model Loading
st.set_page_config(page_title="Face Mask Detector", layout="centered")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_mask_model():
    if os.path.exists("mask_detector.h5"):
        return load_model("mask_detector.h5")
    return None

model = load_mask_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Video Processor Class
class FaceMaskProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Extract and Preprocess Face
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_roi = cv2.resize(face_roi, (128, 128))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            # Prediction
            if model is not None:
                (mask, without_mask) = model.predict(face_roi)[0]
                label = "Mask" if mask > without_mask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                # Draw
                txt = f"{label}: {max(mask, without_mask)*100:.2f}%"
                cv2.putText(img, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        return frame.from_ndarray(img, format="bgr24")

# 3. Streamlit UI
st.title("实时口罩检测系统 (Live Face Mask Detector)")
st.write("This application uses a CNN model to detect if you are wearing a mask in real-time.")

if model is None:
    st.error("Model file 'mask_detector.h5' not found. Please train the model locally first.")
else:
    st.success("Model loaded successfully!")
    
    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox("Choose Mode", ["Live Feed", "About"])

    if mode == "Live Feed":
        st.write("### Webcam Feed")
        webrtc_streamer(
            key="face-mask-detection",
            video_processor_factory=FaceMaskProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.write("### About this project")
        st.info("This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed via Streamlit.")
        st.write("- **Back-end:** TensorFlow, Keras")
        st.write("- **Computer Vision:** OpenCV")
        st.write("- **Web Framework:** Streamlit")

st.markdown("---")
st.caption("Developed for College Mini Project - Face Mask Detection")
