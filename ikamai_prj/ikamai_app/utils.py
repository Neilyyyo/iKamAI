# myapp/utils.py
import cv2
import numpy as np
import tensorflow as tf # CHANGED: Import TensorFlow for TFLite
import mediapipe as mp
import base64
import io
from PIL import Image

class WordPredictor:
    # CHANGED: Updated arguments to accept a single .tflite file instead of json/h5
    def __init__(self, model_path="model.tflite", threshold=0.8):
        
        # CHANGED: Load TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Labels
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

    def mediapipe_detection(self, image, model):
        """Process frame through Mediapipe hand model."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def process_web_frame(self, image_data_base64):
        """
        Receives base64 image from frontend, processes it, updates sequence,
        and returns prediction status.
        """
        try:
            # 1. Decode base64 image
            img_str = image_data_base64.split(',')[1]
            decoded = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(decoded))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Apply the SAME cropping as your original code
            # Note: Ensure the incoming frame is large enough. 
            # If frontend sends 640x480, this crop works.
            cropframe = frame[40:400, 0:300]
            
            # Optional: Visual debug on server (not visible to user)
            # cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

            # 3. Process with Mediapipe
            with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                
                image, results = self.mediapipe_detection(cropframe, hands)
                
                # 4. Extract Keypoints & Update Sequence
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:] # Keep last 30

                # 5. Prediction Logic
                if len(self.sequence) == 30:
                    # CHANGED: TFLite prediction logic
                    
                    # Prepare input data (TFLite usually requires float32)
                    input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                    
                    # Set the input tensor
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    
                    # Run inference
                    self.interpreter.invoke()
                    
                    # Get output tensor
                    res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

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
            
            return self.get_status()

        except Exception as e:
            print(f"Error processing frame: {e}")
            return self.get_status()

    def get_status(self):
        """Return the latest stable prediction + accuracy."""
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