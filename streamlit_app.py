import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import time

# Page config
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("🤚 Hand Gesture Recognition App")
st.write("Detect hand gestures using AI and MediaPipe")

# Sidebar for settings
st.sidebar.title("Settings")
app_mode = st.sidebar.radio("Choose Mode", ["Image Upload", "Real-time Webcam"])

# Load model and scaler (cached for performance)
@st.cache_resource
def load_model_and_scaler():
    try:
        import pickle
        
        # Load model
        model = load_model('models/hand_gesture_model.h5')
        
        # Load the fitted scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training notebook first.")
        st.error("Make sure scaler.pkl and label_encoder.pkl are saved in the folder.")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading files: {e}")
        return None, None, None

# Initialize MediaPipe HandLandmarker
@st.cache_resource
def initialize_hand_landmarker():
    try:
        mp_tasks = mp.tasks
        BaseOptions = mp_tasks.BaseOptions
        HandLandmarker = mp_tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp_tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
        return hand_landmarker, HandLandmarkerOptions, VisionRunningMode, BaseOptions
    except Exception as e:
        st.error(f"⚠️ Error loading MediaPipe: {e}")
        return None, None, None, None

# Function to extract landmarks
def extract_landmarks(hand_landmarks_result):
    """Extract and normalize landmarks from detection result"""
    if not hand_landmarks_result.hand_landmarks:
        return None
    
    landmarks = hand_landmarks_result.hand_landmarks[0]
    
    # Wrist as reference
    wrist = landmarks[0]
    base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
    
    # Extract relative to wrist
    normalized = []
    for lm in landmarks:
        normalized.append(lm.x - base_x)
        normalized.append(lm.y - base_y)
        normalized.append(lm.z - base_z)
    
    # Scale by middle finger distance
    mf_tip = landmarks[12]
    scale = np.sqrt((mf_tip.x - base_x)**2 + (mf_tip.y - base_y)**2 + (mf_tip.z - base_z)**2)
    
    if scale > 0:
        normalized = [v / scale for v in normalized]
    
    return np.array(normalized)

# Function to draw landmarks
def draw_landmarks(image, detection_result):
    """Draw hand landmarks on image"""
    annotated_image = image.copy()
    h, w, c = image.shape
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
            ]
            
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 8, (0, 0, 255), -1)
    
    return annotated_image

# Load model and gesture labels
model, scaler, label_encoder_streamlit = load_model_and_scaler()
hand_landmarker, HandLandmarkerOptions, VisionRunningMode, BaseOptions = initialize_hand_landmarker()

# Gesture classes (from loaded label encoder)
if label_encoder_streamlit is not None:
    gesture_classes = label_encoder_streamlit.classes_
else:
    gesture_classes = ['fist', 'index_finger', 'ok', 'open_palm', 'peace', 'thumb_up']

# ------- IMAGE UPLOAD MODE -------
if app_mode == "Image Upload":
    st.subheader("📸 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert PIL to OpenCV format (RGB to BGR for display)
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            st.warning("Image must be in color format")
            st.stop()
        
        # Detect hand landmarks
        if hand_landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            detection_result = hand_landmarker.detect(mp_image)
            
            # Draw landmarks
            annotated_image = draw_landmarks(image_cv, detection_result)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Original Image with Landmarks")
                st.image(annotated_image_rgb, width='content')
            
            # Make prediction if hand detected
            if detection_result.hand_landmarks:
                landmarks = extract_landmarks(detection_result)
                
                if landmarks is not None and model is not None and scaler is not None:
                    # Normalize landmarks using the fitted scaler
                    landmarks_scaled = scaler.transform([landmarks])
                    
                    # Make prediction
                    prediction = model.predict(landmarks_scaled, verbose=0)
                    confidence = np.max(prediction)
                    predicted_class = np.argmax(prediction)
                    gesture_name = gesture_classes[predicted_class]
                    
                    with col2:
                        st.write("### Prediction Results")
                        st.metric("Detected Gesture", gesture_name.upper(), f"{confidence*100:.1f}%")
                        
                        # Show all predictions
                        st.write("**All Predictions:**")
                        pred_df = pd.DataFrame({
                            'Gesture': gesture_classes,
                            'Confidence': prediction[0] * 100
                        }).sort_values('Confidence', ascending=False)
                        st.bar_chart(pred_df.set_index('Gesture'))
            else:
                st.warning("❌ No hand detected in the image. Try another image!")

# ------- REAL-TIME WEBCAM MODE -------
elif app_mode == "Real-time Webcam":
    st.subheader("🎥 Real-time Hand Gesture Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Click the button below to start webcam detection:")
        start_button = st.button("Start Webcam")
    
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    if start_button:
        st.write("Initializing webcam... (Make sure your webcam is connected)")
        
        # Create placeholders for live updates
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Check if it's connected and allowed.")
        else:
            st.write("Webcam is running! (Press 'Stop' to exit)")
            
            frame_count = 0
            gesture_buffer = []
            stop_webcam = False
            
            st.write("Fist, Index Finger, OK, Open Palm, Peace, Thumb Up")

            # Add a stop button
            stop_button = st.empty()
            stop_webcam_btn = stop_button.button("Stop Webcam")

            
            while True and not stop_webcam_btn:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect landmarks
                if hand_landmarker is not None:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detection_result = hand_landmarker.detect(mp_image)
                    
                    # Draw landmarks
                    annotated_frame = draw_landmarks(frame, detection_result)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Make prediction if hand detected
                    if detection_result.hand_landmarks:
                        landmarks = extract_landmarks(detection_result)
                        
                        if landmarks is not None and model is not None and scaler is not None:
                            landmarks_scaled = scaler.transform([landmarks])
                            prediction = model.predict(landmarks_scaled, verbose=0)
                            confidence = np.max(prediction)
                            predicted_class = np.argmax(prediction)
                            gesture_name = gesture_classes[predicted_class]
                            
                            # Add to buffer for smoothing
                            gesture_buffer.append((gesture_name, confidence))
                            if len(gesture_buffer) > 5:
                                gesture_buffer.pop(0)
                            
                            # Use most common prediction
                            if gesture_buffer and confidence > confidence_threshold:
                                gestures = [g for g, _ in gesture_buffer]
                                most_common = max(set(gestures), key=gestures.count)
                                avg_conf = np.mean([c for _, c in gesture_buffer])
                                
                                # Add text to frame
                                text = f"{most_common.upper()} ({avg_conf*100:.1f}%)"
                                cv2.putText(annotated_frame_rgb, text, (10, 40),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Display frame
                    frame_placeholder.image(annotated_frame_rgb, channels="RGB", width='content')
                    
                    # Update info
                    frame_count += 1
                    if frame_count % 30 == 0:
                        info_placeholder.write(f"Frames processed: {frame_count}")
                else:
                    st.error("MediaPipe not initialized")
                    break
            
            cap.release()
            st.success("✓ Webcam closed")

# Footer
st.divider()
st.markdown("""
### How it works:
1. **Upload Mode**: Upload an image containing a hand gesture
2. **Webcam Mode**: Real-time detection using your webcam
3. The app detects 21 hand landmarks and predicts one of 6 gestures:
   - Fist, Index Finger, OK, Open Palm, Peace, Thumb Up

### Tips:
- Ensure good lighting for better detection
- Keep your hand clearly visible
- Try different angles and distances
""")
