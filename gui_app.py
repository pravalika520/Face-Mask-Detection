import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time

class FaceMaskApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x700")
        self.window.configure(bg="#2c3e50")

        # Load models
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = None
        self.load_mask_model()

        # Variables
        self.video_source = 0
        self.vid = None
        self.is_detecting = False
        self.thread = None

        # UI Elements
        self.create_widgets()

        # Force window to front (Helpful for macOS)
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after_idle(self.window.attributes, '-topmost', False)
        self.window.focus_force()

    def load_mask_model(self):
        if os.path.exists("mask_detector.h5"):
            try:
                self.model = load_model("mask_detector.h5")
                print("[INFO] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Could not load model: {e}")
        else:
            print("[WARNING] mask_detector.h5 not found. Please train the model.")

    def create_widgets(self):
        # Title
        self.title_label = tk.Label(self.window, text="Face Mask Detection System", 
                                   font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="#ecf0f1")
        self.title_label.pack(pady=20)

        # Video Canvas
        self.canvas = tk.Canvas(self.window, width=640, height=480, bg="#34495e", highlightthickness=0)
        self.canvas.pack(pady=10)

        # Button Frame
        self.btn_frame = tk.Frame(self.window, bg="#2c3e50")
        self.btn_frame.pack(pady=20)

        # Start Button
        self.start_btn = tk.Button(self.btn_frame, text="Start Detection", width=15, 
                                  command=self.start_detection, bg="#27ae60", fg="black", 
                                  font=("Helvetica", 12, "bold"), relief="flat")
        self.start_btn.grid(row=0, column=0, padx=10)

        # Stop Button
        self.stop_btn = tk.Button(self.btn_frame, text="Stop Detection", width=15, 
                                 command=self.stop_detection, bg="#c0392b", fg="black", 
                                 font=("Helvetica", 12, "bold"), relief="flat", state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10)

        # Train Button (Helper)
        self.train_btn = tk.Button(self.btn_frame, text="Train Model", width=15, 
                                  command=self.run_training, bg="#2980b9", fg="black", 
                                  font=("Helvetica", 12, "bold"), relief="flat")
        self.train_btn.grid(row=0, column=2, padx=10)

        # Footer / Status
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = tk.Label(self.window, textvariable=self.status_var, 
                                    font=("Helvetica", 10), bg="#2c3e50", fg="#bdc3c7")
        self.status_label.pack(side="bottom", pady=10)

    def start_detection(self):
        if self.model is None:
            messagebox.showerror("Error", "Model file 'mask_detector.h5' not found! Please train the model first.")
            return

        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.is_detecting = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Status: Detecting...")
        
        self.update_frame()

    def stop_detection(self):
        self.is_detecting = False
        if self.vid:
            self.vid.release()
        self.canvas.delete("all")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Stopped")

    def run_training(self):
        if messagebox.askyesno("Train Model", "This will start the training process. It might take a while. Continue?"):
            self.status_var.set("Status: Training...")
            threading.Thread(target=self._execute_training, daemon=True).start()

    def _execute_training(self):
        os.system("python train_model.py")
        self.load_mask_model()
        self.window.after(0, lambda: self.status_var.set("Status: Training Complete. Model Loaded."))
        self.window.after(0, lambda: messagebox.showinfo("Success", "Training complete and model loaded!"))

    def update_frame(self):
        if not self.is_detecting:
            return

        ret, frame = self.vid.read()
        if ret:
            # Face Detection Logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_roi = cv2.resize(face_roi, (128, 128))
                face_roi = face_roi.astype("float32") / 255.0
                face_roi = img_to_array(face_roi)
                face_roi = np.expand_dims(face_roi, axis=0)

                (mask, without_mask) = self.model.predict(face_roi)[0]
                label = "Mask" if mask > without_mask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (255, 0, 0) # Tkinter uses RGB for drawing sometimes, but CV2 is BGR. 
                # Note: For drawing on the frame we use CV2 colors (BGR)
                cv2_color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                txt = f"{label}: {max(mask, without_mask)*100:.2f}%"
                cv2.putText(frame, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2_color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), cv2_color, 2)

            # Convert to RGB for PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.is_detecting:
            self.window.after(10, self.update_frame)

    def __del__(self):
        if self.vid and self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMaskApp(root, "Face Mask Detector GUI")
    root.mainloop()
