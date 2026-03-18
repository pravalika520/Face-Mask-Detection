import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# 1. Load Model and Face Detector
print("[INFO] Loading face detector and mask detector model...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if model exists
if not os.path.exists("mask_detector.h5"):
    print("[ERROR] 'mask_detector.h5' not found. Please train the model first using train_model.py")
    exit()

model = load_model("mask_detector.h5")

# 2. Access Webcam
print("[INFO] Starting webcam feed...")
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Preprocess image for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess for CNN model
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = cv2.resize(face_roi, (128, 128))
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Make prediction
        (mask, without_mask) = model.predict(face_roi)[0]

        # Determine label and color
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green for Mask, Red for No Mask
        
        # Label with probability
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        # Draw bounding box and label
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the output frame
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("[INFO] Stopping webcam feed...")
cap.release()
cv2.destroyAllWindows()
