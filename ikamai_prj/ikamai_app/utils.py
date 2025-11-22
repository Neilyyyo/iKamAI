# myapp/utils.py
import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp

class WordPredictor:
    def __init__(self, model_json="model.json", model_weights="model.h5", threshold=0.8):
        # Load trained model
        with open(model_json, "r") as f:
            model_data = f.read()
        self.model = model_from_json(model_data)
        self.model.load_weights(model_weights)

        # Labels (use your trained labels)
        self.actions = np.array([
            'hello', 'yes', 'no', 'thanks',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eat', 'wrong', 'sorry', 'like', 'iloveyou','ihateyou', 'eat'
        ])

        # State variables
        self.threshold = threshold
        self.sequence = []      # last 30 frames of keypoints
        self.sentence = []      # recognized words
        self.accuracy = []      # confidence values
        self.predictions = []   # raw predictions (indices)

        # Mediapipe setup
        self.mp_hands = mp.solutions.hands

        # Camera placeholder
        self.cap = None

    def reset(self):
        """Reset all state."""
        self.sequence = []
        self.sentence = []
        self.accuracy = []
        self.predictions = []

    def extract_keypoints(self, results):
        if results.multi_hand_landmarks:
            # take only the first detected hand
            hand = results.multi_hand_landmarks[0]
            return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
        else:
            return np.zeros(21 * 3)  # 21 landmarks * 3 coordinates



    def generate_frames(self):
        """Capture webcam frames and yield them for streaming."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        try:
            with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Crop region for detection
                    cropframe = frame[40:400, 0:300]
                    frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

                    # Run Mediapipe
                    image, results = self.mediapipe_detection(cropframe, hands)

                    # Extract keypoints
                    keypoints = self.extract_keypoints(results)
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-30:]

                    try:
                        if len(self.sequence) == 30:
                            # Make prediction
                            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                            self.predictions.append(np.argmax(res))

                            # Check consistency + confidence
                            if np.unique(self.predictions[-15:])[0] == np.argmax(res):
                                if res[np.argmax(res)] > self.threshold:
                                    pred_word = self.actions[np.argmax(res)]
                                    confidence = round(res[np.argmax(res)] * 100, 2)

                                    if len(self.sentence) > 0:
                                        if pred_word != self.sentence[-1]:
                                            self.sentence.append(pred_word)
                                            self.accuracy.append(confidence)
                                    else:
                                        self.sentence.append(pred_word)
                                        self.accuracy.append(confidence)

                            if len(self.sentence) > 1:
                                # keep only the most recent
                                self.sentence = self.sentence[-1:]
                                self.accuracy = self.accuracy[-1:]
                    except Exception:
                        pass

                    # Encode frame for streaming
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            self.release()

    def release(self):
        """Release camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def mediapipe_detection(self, image, model):
        """Process frame through Mediapipe hand model."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def get_status(self):
        """Return the latest stable prediction + accuracy (like test script)."""
        if self.sentence and self.accuracy:
            return {
                'prediction': self.sentence[-1],
                'accuracy': f"{self.accuracy[-1]}%"
            }
        else:
            return {
                'prediction': '',
                'accuracy': ''
            }
