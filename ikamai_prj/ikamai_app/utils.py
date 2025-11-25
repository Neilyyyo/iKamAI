import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter

import mediapipe as mp
import base64
import io
from PIL import Image

class WordPredictor:
    def __init__(self, model_path="model.tflite", threshold=0.8):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.actions = np.array([
            'hello', 'yes', 'no', 'thanks',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eat', 'wrong', 'sorry', 'like', 'iloveyou','ihateyou', 'eat'
        ])

        self.threshold = threshold
        self.sequence = []
        self.sentence = []
        self.accuracy = []
        self.predictions = []

        self.mp_hands = mp.solutions.hands
        
        # --- FIX: Initialize Hands HERE, not in the loop ---
        self.hands = self.mp_hands.Hands(
            model_complexity=0, # Keep 0 for speed on server
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # ---------------------------------------------------

    def reset(self):
        self.sequence = []
        self.sentence = []
        self.accuracy = []
        self.predictions = []

    def extract_keypoints(self, results):
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
        else:
            return np.zeros(21 * 3)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def process_web_frame(self, image_data_base64):
        try:
            img_str = image_data_base64.split(',')[1]
            decoded = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(decoded))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cropframe = frame[40:400, 0:300]

            # --- FIX: Use the self.hands instance we created in __init__ ---
            image, results = self.mediapipe_detection(cropframe, self.hands)
            # ---------------------------------------------------------------

            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:
                input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                self.predictions.append(np.argmax(res))

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
                    self.sentence = self.sentence[-1:]
                    self.accuracy = self.accuracy[-1:]
            
            return self.get_status()

        except Exception as e:
            print(f"Error processing frame: {e}")
            return self.get_status()

    def get_status(self):
        if self.sentence and self.accuracy:
            return {
                'prediction': self.sentence[-1],
                'accuracy': f"{self.accuracy[-1]}%"
            }
        else:
            return {
                'prediction': 'None',
                'accuracy': '0%'
            }