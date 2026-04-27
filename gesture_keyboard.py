import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import keyboard
from pynput.mouse import Controller as MouseController, Button
import time
from collections import deque

# Initialize controllers
mouse = MouseController()

# Gesture to Key Mapping (customize as needed)
# Use 'click' for lefst click, 'rclick' for right click, None to do nothing
GESTURE_TO_KEY = {
    'fist': 'w',           
    'index_finger': 'a',   
    'peace': 'd',          
    'thumb_up': 'rclick',   
    'open_palm': ' ',      
    'ok': 'click',          
    'unrecognized': None    
}

class GestureKeyboardMapper:
    def __init__(self):
        """Initialize the gesture recognition and keyboard mapping system"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hand_landmarker = None
        self.running = False
        self.listening = False
        self.last_gesture = 'None'
        self.gesture_buffer = deque(maxlen=5)
        self.last_key_press = {}  # Prevent spam pressing
        self.key_cooldown = 0.1  # 100ms cooldown between presses
        
        # Track currently held key
        self.currently_held_gesture = None
        self.currently_held_key = None
        self.keep_pressing = False  # Flag to continuously press key
        self.press_thread = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained model, scaler, and label encoder"""
        try:
            print("Loading gesture recognition model...")
            self.model = load_model('models/hand_gesture_model.h5')
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✓ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model files not found! Please run the training notebook first.")
            print("   Expected files: models/hand_gesture_model.h5, models/scaler.pkl, models/label_encoder.pkl")
            exit(1)
    
    def initialize_hand_landmarker(self):
        """Initialize MediaPipe HandLandmarker"""
        try:
            print("Initializing MediaPipe HandLandmarker...")
            mp_tasks = mp.tasks
            BaseOptions = mp_tasks.BaseOptions
            HandLandmarker = mp_tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp_tasks.vision.RunningMode
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE
            )
            self.hand_landmarker = HandLandmarker.create_from_options(options)
            print("✓ HandLandmarker initialized!")
        except Exception as e:
            print(f"❌ Error loading MediaPipe: {e}")
            exit(1)
    
    def extract_landmarks(self, hand_landmarks_result):
        """Extract and normalize landmarks from detection result"""
        if not hand_landmarks_result.hand_landmarks:
            return None
        
        landmarks = hand_landmarks_result.hand_landmarks[0]
        wrist = landmarks[0]
        base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
        
        normalized = []
        for lm in landmarks:
            normalized.append(lm.x - base_x)
            normalized.append(lm.y - base_y)
            normalized.append(lm.z - base_z)
        
        mf_tip = landmarks[12]
        scale = np.sqrt((mf_tip.x - base_x)**2 + (mf_tip.y - base_y)**2 + (mf_tip.z - base_z)**2)
        
        if scale > 0:
            normalized = [v / scale for v in normalized]
        
        return np.array(normalized)
    
    def detect_gesture(self, frame):
        """Detect gesture from video frame"""
        if self.hand_landmarker is None:
            return 'None'
        
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_landmarker.detect(mp_image)
            
            if detection_result.hand_landmarks:
                landmarks = self.extract_landmarks(detection_result)
                
                if landmarks is not None:
                    landmarks_scaled = self.scaler.transform([landmarks])
                    prediction = self.model.predict(landmarks_scaled, verbose=0)
                    predicted_class = np.argmax(prediction)
                    gesture = self.label_encoder.classes_[predicted_class]
                    
                    self.gesture_buffer.append(gesture)
                    if len(self.gesture_buffer) > 0:
                        # Require 3+ consistent detections out of 5
                        gesture_counts = {}
                        for g in self.gesture_buffer:
                            gesture_counts[g] = gesture_counts.get(g, 0) + 1
                        
                        # Find most common gesture
                        most_common = max(gesture_counts, key=gesture_counts.get)
                        confidence = gesture_counts[most_common] / len(self.gesture_buffer)
                        
                        # Only return gesture if confidence is high
                        if confidence >= 0.6:  # 3 out of 5 frames
                            return most_common
        except Exception as e:
            pass
        
        return 'None'
    
    def continuous_press_thread(self, key):
        """Thread that continuously holds a key down"""
        try:
            keyboard.press(key)
            print(f"  → Key '{key}' PRESSED and HELD")
            while self.keep_pressing and self.running:
                time.sleep(0.01)  # Keep key held
        except Exception as e:
            print(f"Error during press: {e}")
        finally:
            try:
                keyboard.release(key)
                print(f"  → Key '{key}' RELEASED")
            except:
                pass
    
    def hold_key(self, gesture):
        """Hold key continuously until gesture changes"""
        if gesture not in GESTURE_TO_KEY:
            return
        
        key = GESTURE_TO_KEY[gesture]
        
        # Skip if key is None (do nothing for this gesture)
        if key is None:
            return
        
        # If same gesture, keep holding
        if gesture == self.currently_held_gesture:
            return
        
        # Stop previous key pressing if any
        self.keep_pressing = False
        if self.press_thread is not None and self.press_thread.is_alive():
            self.press_thread.join(timeout=0.3)
        
        # Press new key
        try:
            if key == 'click':
                # Single click, don't hold
                mouse.click(Button.left, 1)
                self.currently_held_gesture = None
                self.currently_held_key = None
                self.keep_pressing = False
                print(f"🖱️  {gesture.upper()} → LEFT CLICK")
            elif key == 'rclick':
                # Single right click, don't hold
                mouse.click(Button.right, 1)
                self.currently_held_gesture = None
                self.currently_held_key = None
                self.keep_pressing = False
                print(f"🖱️  {gesture.upper()} → RIGHT CLICK")
            else:
                # Start continuous pressing in separate thread
                self.currently_held_gesture = gesture
                self.currently_held_key = key
                self.keep_pressing = True
                self.press_thread = threading.Thread(target=self.continuous_press_thread, args=(key,), daemon=True)
                self.press_thread.start()
                print(f"🔑 {gesture.upper()} → HOLDING '{key}'")
        except Exception as e:
            print(f"Error: {e}")
    
    def release_all_keys(self):
        """Release any currently held keys"""
        self.keep_pressing = False
        if self.press_thread is not None and self.press_thread.is_alive():
            self.press_thread.join(timeout=0.3)
        
        if self.currently_held_key is not None:
            try:
                if self.currently_held_key != 'click':
                    keyboard.release(self.currently_held_key)
                    print(f"🔓 RELEASED ('{self.currently_held_key}')")
            except:
                pass
            finally:
                self.currently_held_gesture = None
                self.currently_held_key = None
                self.press_thread = None
    
    def run_detection_loop(self):
        """Main detection loop running in background thread"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not access webcam!")
            return
        
        print("✓ Webcam opened. Starting gesture detection...")
        
        frame_count = 0
        no_detection_frames = 0
        grace_period = 10  # Frames to wait before releasing (at 30fps = ~330ms)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            if self.listening:
                gesture = self.detect_gesture(frame)
                
                if gesture != 'None' and gesture != self.currently_held_gesture:
                    # NEW gesture detected - switch keys
                    self.last_gesture = gesture
                    self.hold_key(gesture)
                    no_detection_frames = 0
                elif gesture != 'None':
                    # Same gesture - keep holding
                    self.last_gesture = gesture
                    no_detection_frames = 0
                else:
                    # No gesture detected
                    no_detection_frames += 1
                    
                    # If we've had no detection for grace_period, release
                    if no_detection_frames >= grace_period:
                        if self.currently_held_key is not None:
                            self.release_all_keys()
                        self.last_gesture = 'unrecognized'
            else:
                # If listening is toggled off, release all keys
                if self.currently_held_key is not None:
                    self.release_all_keys()
            
            # Display frame with gesture info
            h, w, c = frame.shape
            cv2.putText(frame, f"Status: {'LISTENING' if self.listening else 'PAUSED'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.listening else (0, 0, 255), 2)
            cv2.putText(frame, f"Gesture: {self.last_gesture}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            held_status = f"Holding: {self.currently_held_gesture}" if self.currently_held_gesture else "None"
            cv2.putText(frame, held_status, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(frame, "Press 0 to toggle, Q to quit", 
                       (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Gesture Keyboard Mapper", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('0'):
                self.listening = not self.listening
                status = "LISTENING" if self.listening else "PAUSED"
                print(f"\n→ {status}\n")
            
            frame_count += 1
        
        # Clean up: release any held keys
        self.release_all_keys()
        cap.release()
        cv2.destroyAllWindows()
    
    def start(self):
        """Start the gesture keyboard mapper"""
        print("\n" + "="*60)
        print("🎮 GESTURE KEYBOARD MAPPER")
        print("="*60)
        print("\nGesture → Action Mapping:")
        for gesture, key in GESTURE_TO_KEY.items():
            if key is None:
                print(f"  {gesture:15} → (do nothing)")
            elif key == 'click':
                print(f"  {gesture:15} → LEFT CLICK")
            elif key == 'rclick':
                print(f"  {gesture:15} → RIGHT CLICK")
            else:
                print(f"  {gesture:15} → '{key}'")
        
        print("\n" + "-"*60)
        print("Controls:")
        print("  0      = Toggle listening ON/OFF")
        print("  Q      = Quit application")
        print("-"*60 + "\n")
        print("⚠️  This script requires admin/elevated privileges to work!")
        print("   On Windows: Run Command Prompt as Administrator\n")
        
        self.initialize_hand_landmarker()
        self.running = True
        self.listening = True
        
        try:
            self.run_detection_loop()
        except KeyboardInterrupt:
            print("\n\n✓ Exiting gracefully...")
        finally:
            self.running = False
            self.release_all_keys()
            print("✓ Gesture keyboard mapper stopped!")

def main():
    """Main entry point"""
    mapper = GestureKeyboardMapper()
    mapper.start()

if __name__ == "__main__":
    main()
