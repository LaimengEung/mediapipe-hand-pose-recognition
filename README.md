# Hand Gesture Recognition with MediaPipe & Neural Networks

A comprehensive machine learning project for real-time hand gesture recognition using MediaPipe hand landmarks and multiple deep learning & ensemble models.

## 🎯 Project Overview

This project implements a complete pipeline for gesture recognition that:
- **Extracts hand landmarks** using MediaPipe's pre-trained hand detection model (21 joints per hand)
- **Normalizes features** with StandardScaler for consistent neural network input
- **Trains & compares 4 different models** to validate architectural choices
- **Achieves 99%+ accuracy** across all models tested
- **Provides real-time gesture control** for keyboard and mouse input
- **Supports web interface** via Streamlit for easy interaction

## 📊 Dataset

- **Total Samples**: ~9,092 across 7 gesture classes
- **Features**: 63 normalized hand landmark coordinates (21 joints × 3 dimensions: x, y, z)
- **Gesture Classes**: 
  - Fist
  - Index Finger
  - OK
  - Open Palm
  - Peace
  - Thumb Up
  - Unrecognized
- **Split**: 70% training, 15% validation, 15% test (stratified)

## 🤖 Models Compared

### Model 1: Random Forest ⭐ FASTEST
- **Architecture**: 100 trees, max_depth=15
- **Test Accuracy**: ~99.512%
- **Training Time**: ~0-2s 🚀
- **Pros**: Lightning-fast, interpretable, no overfitting
- **Cons**: Less flexible than neural networks

### Model 2: XGBoost ⭐ FASTEST
- **Architecture**: 100 estimators, max_depth=7, learning_rate=0.1
- **Test Accuracy**: ~98.75%
- **Training Time**: ~2-7s 🚀
- **Pros**: fast, gradient boosted trees are powerful
- **Cons**: More complex hyperparameter tuning

### Model 3: Simple MLP ⭐ DEPLOYMENT CHOICE
- **Architecture**: Input(63) → Dense(128, relu) → Dropout(0.3) → Dense(7, softmax)
- **Test Accuracy**: 99.56% 🏆
- **Training Time**: ~20-30 seconds
- **Pros**: Excellent accuracy, balanced speed/performance
- **Cons**: Slower than ensemble methods

### Model 4: Deeper MLP
- **Architecture**: Input(63) → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dropout(0.2) → Dense(7, softmax)
- **Test Accuracy**: ~99.41%
- **Training Time**: ~30-40 seconds
- **Pros**: Still highly accurate
- **Cons**: No significant improvement over Simple DNN, longer training

## 📈 Key Findings

| Model | Accuracy | Training Time | Speed Rank | Recommendation |
|-------|----------|---------------|-----------|-----------------|
| Simple MLP | **99.56%** | 20-30s | 3rd | ✅ **Selected** |
| Deeper MLP | 99.41% | 30-40s | 4th | Not needed |
| Random Forest | 99.12% | 0-2s | 1st | Great alternative |
| XGBoost | 98.75% | 2-7s | 2nd | Great alternative |

### **Key Insight**: Feature Quality > Model Complexity
All models achieve 99%+ accuracy because **MediaPipe hand landmarks provide exceptional feature quality**. The extracted 21 joints (63 coordinates) contain rich gesture information, making even simple models highly effective.

## 🗂️ Project Structure

```
mediapipe-final-project/
├── training_model.ipynb           # Complete training pipeline & model comparison
├── streamlit_app.py               # Web UI for gesture recognition
├── gesture_keyboard.py            # Real-time gesture → keyboard/mouse control
├── hand_landmarker.task           # Pre-trained MediaPipe hand detection model
├── models/
│   ├── hand_gesture_model.h5      # Saved Simple MLP model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   └── label_encoder.pkl          # Gesture label encoder
├── results/                       # Auto-saved visualization outputs
├── data_collection/               # Data collection & preprocessing utilities
│   ├── collect_data.py            # Collect gesture samples from webcam
│   ├── clean_and_merge.py         # Clean and merge collected datasets
│   └── final_dataset.csv          # Processed dataset ready for training
├── dataset/
│   └── merged_dataset_cleaned.csv # Combined training data
└── README.md                      # This file
```

## 🚀 Installation & Setup

### Requirements
```bash
pip install tensorflow keras
pip install scikit-learn xgboost
pip install mediapipe opencv-python
pip install streamlit
pip install keyboard pynput pyautogui
```

### Quick Start

**1. Train Models**
```bash
# Open training_model.ipynb in Jupyter
# Run all cells to train and compare models
# Models will be saved to ./models/

> OR use the pre-trained models in ./models/
```

**2. Web Interface (Streamlit)**
```bash
streamlit run streamlit_app.py
```
- Upload images or use webcam for real-time gesture recognition
- Set confidence threshold
- View predictions with probabilities

**3. Real-time Gesture Control**
```bash
python gesture_keyboard.py
```
- Press `0` to toggle gesture listening ON/OFF
- Press `Q` to quit
- Control keyboard/mouse with hand gestures:
  - **Fist** → 'w' (forward)
  - **Index Finger** → 'a' (left)
  - **Peace** → 'd' (right)
  - **OK** → left mouse click
  - **Thumb Up** → right mouse click
  - **Open Palm** → space bar

> You may change this to any other keybinds.

## 📋 Training Pipeline (training_model.ipynb)

### Section 1: Dataset Overview
- Load `merged_dataset_cleaned.csv`
- Visualize class distribution
- Explore gesture samples

### Section 2: Data Preparation
- Label encoding (gesture names → numeric values)
- Feature normalization (StandardScaler)
- Train/validation/test split (70-15-15)

### Section 3: MediaPipe Integration
- Download `hand_landmarker.task` from Google
- Initialize HandLandmarker with IMAGE running mode
- Visualize hand landmarks for each gesture class

### Section 4: Model Training & Comparison
- **Model 1**: Train Random Forest
- **Model 2**: Train XGBoost
- **Model 3**: Train Simple MLP (1 hidden layer)
- **Model 4**: Train Deeper MLP (3 hidden layers)

### Section 5: Model Comparison & Analysis
- Summary accuracy table
- Training time comparison
- Visualization charts (accuracy vs time)
- Classification reports for best model

### Selected: Simple MLP 
```
Input Layer:       63 features (hand landmarks)
                   ↓
Hidden Layer:      128 neurons (ReLU activation)
                   ↓
Dropout:           30% dropout for regularization
                   ↓
Output Layer:      7 neurons (softmax for 7 gesture classes)

Training Config:
- Optimizer: Adam (learning_rate=0.001)
- Loss: sparse_categorical_crossentropy
- Metrics: accuracy
- Epochs: 40 (with early stopping, patience=5)
- Batch Size: 16
```

## 📚 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hand Detection | MediaPipe Tasks API | Extract 21 hand landmarks (x,y,z) |
| Feature Extraction | MediaPipe | Normalized coordinates (0-1 range) |
| Data Scaling | scikit-learn StandardScaler | Normalize features to mean=0, std=1 |
| Neural Networks | TensorFlow/Keras | Simple & Deeper DNN models |
| Ensemble Models | scikit-learn, XGBoost | Random Forest & Gradient Boosting |
| Web Interface | Streamlit | Easy deployment & interaction |
| Gesture Control | keyboard, pynput | Send keyboard/mouse commands |
| Video Processing | OpenCV | Real-time webcam capture |

## 📊 Model Comparison Insights

**1. Feature Quality Matters Most**
- All models achieve 99%+ accuracy
- MediaPipe landmarks provide rich, discriminative features
- Model complexity doesn't guarantee better results

**2. Speed Tradeoff**
- Ensemble methods (RF, XGBoost): ~0.01-0.02s per inference
- Neural Networks: ~0.1-0.2s per inference
- For real-time use: Neural networks acceptable at 10-15 FPS

**3. Production Considerations**
- **Simple MLP**: Best overall choice (accuracy + inference speed)
- **Random Forest**: Best for deployment (fastest inference)
- **XGBoost**: Good alternative (fast + interpretable)
- **Deeper MLP**: Not needed (minimal accuracy gain, slower)

## 🎮 Applications

1. **Gesture-Based Control**: Control presentations, games, applications
2. **Accessibility**: Hand gesture interfaces for users with limited keyboard/mouse access
3. **Sign Language Recognition**: Foundation for ASL/other sign language detection
4. **Interactive Art**: Real-time gesture-triggered art installations
5. **Gaming**: Gesture-controlled game experiences
6. **Rehabilitation**: Hand tracking for physical therapy monitoring

## 🔮 Future Improvements

- [ ] Add more gesture classes (thumbs down, peace signs variations, etc.)
- [ ] Implement motion-based gestures (swipe, circle, wave)
- [ ] Add confidence scoring & temporal smoothing
- [ ] Multi-hand support (both left and right hands)
- [ ] Real-time performance optimization
- [ ] Deploy as lightweight edge model (TensorFlow Lite)
- [ ] Add sign language recognition pipeline
- [ ] Implement transfer learning for domain adaptation

## 📝 Files Description

- **training_model.ipynb**: Complete ML pipeline with model comparison
- **streamlit_app.py**: Web UI for gesture recognition via image upload/webcam
- **gesture_keyboard.py**: Real-time gesture detection with keyboard/mouse control
- **hand_landmarker.task**: Pre-trained MediaPipe model for hand detection

## 🤝 Contributing

This project was developed as part of ITM-360 (AI) course at American University of Phnom Penh (AUPP).

Key learning outcomes:
- Feature engineering importance (MediaPipe landmarks)
- Model comparison methodology
- Deployment considerations (accuracy vs speed)
- Real-time computer vision applications
- Full ML pipeline (training → evaluation → deployment)
