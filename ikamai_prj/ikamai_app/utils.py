# utils.py
import cv2
import numpy as np
import mediapipe as mp
import os
import base64
import threading
import tensorflow as tf  # Changed: Import TF for TFLite Interpreter

# --- 1. Keypoints Extraction Logic (UNCHANGED) ---

def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def image_process(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
        
    return np.concatenate([lh, rh])

# --- 2. Word Predictor Class (Updated for TFLite) ---

class WordPredictor:
    def __init__(self, model_path, actions_list):
        print(f"Loading TFLite Word Model from: {model_path}")
        
        # --- CHANGED: Load TFLite Interpreter instead of Keras model ---
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # ---------------------------------------------------------------

        self.actions = np.array(actions_list)
        
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.70, 
            min_tracking_confidence=0.70
        )
        
        self.lock = threading.Lock()
        
        # Prediction Variables
        self.sequence = []
        self.current_text = ""
        self.last_prediction = None
        
        # Thresholds
        self.threshold = 0.98
        self.frames_required = 20
        
        # Stabilization Variables
        self.hand_present = False
        self.skip_counter = 0
        self.SKIP_FRAMES = 2 
        
    def process_web_frame(self, base64_image):
        result_data = {
            "status": "error",
            "message": "Processing failed",
            "current_word": self.current_text,
            "prediction_made": False
        }

        try:
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            image_bytes = base64.b64decode(base64_image)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Failed to decode image"}
        except Exception as e:
            return {"status": "error", "message": f"Decode error: {str(e)}"}

        with self.lock: 
            try:
                results = image_process(img, self.holistic)
                
                # Check for Hands
                hand_detected = results.left_hand_landmarks or results.right_hand_landmarks
                confidence = 0.0
                prediction_made = False

                if hand_detected:
                    # --- Stabilization Logic Start ---
                    if not self.hand_present:
                        self.hand_present = True
                        self.skip_counter = self.SKIP_FRAMES
                    
                    if self.skip_counter > 0:
                        self.skip_counter -= 1
                        return {
                            "status": "stabilizing", 
                            "current_word": self.current_text,
                            "prediction_made": False
                        }
                    # --- Stabilization Logic End ---

                    # Extract Keypoints
                    keypoints = keypoint_extraction(results)
                    self.sequence.append(keypoints)
                    
                    if len(self.sequence) == self.frames_required:
                        # --- CHANGED: TFLite Inference Logic ---
                        
                        # 1. Prepare input: Expand dims AND ensure float32 (TFLite requirement)
                        input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                        
                        # 2. Set tensor
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                        
                        # 3. Invoke interpreter
                        self.interpreter.invoke()
                        
                        # 4. Get result
                        res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                        # ----------------------------------------

                        confidence = np.max(res)
                        
                        if confidence >= self.threshold:
                            predicted_index = np.argmax(res)
                            predicted_action = self.actions[predicted_index]
                            
                            if predicted_action != self.last_prediction:
                                self.current_text = predicted_action
                                self.last_prediction = predicted_action
                                prediction_made = True
                        
                        self.sequence = [] # Reset after prediction attempt

                else:
                    self.hand_present = False
                    self.sequence = []

                result_data = {
                    "status": "success",
                    "current_word": self.current_text,
                    "confidence": float(confidence),
                    "frames_captured": len(self.sequence),
                    "prediction_made": prediction_made
                }

            except Exception as e:
                print(f"Error in processing: {e}")
                result_data["message"] = str(e)

        return result_data

    def reset(self):
        with self.lock:
            self.sequence = []
            self.current_text = ""
            self.last_prediction = None
            self.hand_present = False
            self.skip_counter = 0