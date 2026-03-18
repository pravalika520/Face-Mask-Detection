import os
import cv2
import numpy as np

def create_dummy_dataset():
    """
    Creates a small dummy dataset of colored squares to test the training pipeline.
    In a real scenario, you should replace these with actual face images.
    """
    base_dir = "dataset"
    categories = ["with_mask", "without_mask"]
    
    print("[INFO] Creating dummy dataset for testing...")
    
    for category in categories:
        path = os.path.join(base_dir, category)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[INFO] Created directory: {path}")
        
        # Create 50 dummy images for each category
        for i in range(50):
            # Create a 128x128 image
            image = np.zeros((128, 128, 3), dtype="uint8")
            
            if category == "with_mask":
                # Green square for 'with_mask' simulation
                cv2.rectangle(image, (20, 20), (108, 108), (0, 255, 0), -1)
            else:
                # Red square for 'without_mask' simulation
                cv2.rectangle(image, (20, 20), (108, 108), (0, 0, 255), -1)
            
            file_path = os.path.join(path, f"sample_{i}.jpg")
            cv2.imwrite(file_path, image)
            
    print("[INFO] Dummy dataset created successfully in 'dataset/' folder.")
    print("[TIP] For a real project, please replace these images with actual photos of people with and without masks.")

if __name__ == "__main__":
    create_dummy_dataset()
