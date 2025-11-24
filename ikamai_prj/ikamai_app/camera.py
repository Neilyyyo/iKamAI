import cv2
import os

class VideoCamera:
    def __init__(self):
        # Check if running on Render (Render sets the 'RENDER' env var automatically)
        if os.environ.get('RENDER'):
            print("Render detected: Camera initialization skipped to prevent crash.")
            self.cap = None
        else:
            # Only try to open the camera if we are NOT on the server
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Warning: Camera could not be opened locally.")
                    self.cap = None
            except Exception as e:
                print(f"Error opening camera: {e}")
                self.cap = None

    def get_frame(self):
        # If the camera is not initialized (Server side), return None
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame  # return raw frame (not encoded)

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()