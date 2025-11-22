# camera.py
import cv2

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame  # return raw frame (not encoded)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
