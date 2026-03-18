# Face Mask Detection Project

This project implements a real-time face mask detection system using a Convolutional Neural Network (CNN) and OpenCV.

## Features
- Detects human faces in a live webcam feed.
- Classifies each face as **Mask** (Green) or **No Mask** (Red).
- Displays real-time confidence scores and FPS.
- Supports multiple faces in a single frame.

## Project Structure
```text
facemask detection/
├── dataset/
│   ├── with_mask/        # Images of people wearing masks
│   └── without_mask/     # Images of people not wearing masks
├── train_model.py        # Script to train the CNN model
├── live_mask_detection.py# Script for real-time webcam detection
├── gui_app.py            # Desktop GUI for mask detection
├── download_sample_data.py # Script to create dummy data for testing
├── mask_detector.h5      # Trained model (generated after training)
├── plot.png              # Training accuracy/loss plot (generated)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── haarcascade_frontalface_default.xml # OpenCV face detection model
```

## Installation

1.  **Clone the repository or download the files.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Training the Model
Before running the live detection, you need to train the model. 

#### Option A: Use your own dataset
Ensure you have your dataset organized in the `dataset/` folder with `with_mask` and `without_mask` subdirectories.

#### Option B: Use dummy data (for testing)
If you don't have a dataset yet, run this script to create a dummy one:
```bash
python download_sample_data.py
```

After either option, run the training script:
```bash
python train_model.py
```
This will generate `mask_detector.h5` and `plot.png`.

### 2. Live Webcam Detection
You can run the detection using the terminal script or the new GUI.

#### Option A: Terminal Script
```bash
python live_mask_detection.py
```

#### Option C: Web Interface (Streamlit)
```bash
python -m streamlit run streamlit_app.py
```

## Deployment to Web (Free)

To get a shareable link for your project:

1.  **Upload to GitHub:** Create a new repository and upload all your project files (including `mask_detector.h5`).
2.  **Connect to Streamlit Cloud:**
    - Go to [share.streamlit.io](https://share.streamlit.io).
    - Sign in with GitHub.
    - Click **"New app"** and select your repository and `streamlit_app.py`.
3.  **Deploy:** Click **"Deploy!"**. You will get a unique URL to share with others.

- Press **'q'** (in terminal mode) or click **'Stop'** (in GUI/Web mode) to quit.

## Troubleshooting (macOS)

If you see `OpenCV: not authorized to capture video`, follow these steps:

1.  **Grant Camera Access:**
    - Go to **System Settings** > **Privacy & Security** > **Camera**.
    - Ensure your **Terminal** (or IDE like VS Code) is toggled **ON**.
2.  **Restart Terminal:**
    - Close the terminal completely and reopen it after granting permission.
3.  **Try different camera index:**
    - If you have multiple cameras, edit `live_mask_detection.py` and change `cv2.VideoCapture(0)` to `1` or `2`.
