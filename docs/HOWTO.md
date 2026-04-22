# How to Use This Project

## ⚙️ Installation

### 1. Install Python Packages
```bash
pip install -r requirements.txt
```

### 2. Download Hand Landmarker Model
The model downloads automatically, but if it fails:
```bash
# Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# Save to: ./hand_landmarker.task
```

## 🔧 Getting Started

### Option 1: Train Your Own Model
```bash
# Open training_model.ipynb in Jupyter Notebook
jupyter notebook training_model.ipynb

# Run all cells (takes ~10-15 minutes)
# Models save to ./models/ automatically
```

### Option 2: Use Web Interface
```bash
streamlit run streamlit_app.py
```
- Opens in browser at `http://localhost:8501`
- Upload images or use webcam
- Adjust confidence threshold

### Option 3: Real-time Gesture Control
```bash
python gesture_keyboard.py
```
- Press `0` to toggle listening
- Press `Q` to quit
- Gesture → keyboard/mouse command mapping

## ❓ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Solution: Install TensorFlow
pip install tensorflow --upgrade
```

### Issue: "No module named 'mediapipe'"
```bash
# Solution: Install MediaPipe
pip install mediapipe
```

### Issue: "hand_landmarker.task not found"
- The file downloads automatically on first run
- If it fails, download manually from [Google Storage](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- Save to project root directory

### Issue: "Permission denied" when running gesture_keyboard.py
- On Windows: Run as Administrator
- On Mac/Linux: May need `sudo` or check accessibility permissions

### Issue: Webcam not working in Streamlit
```bash
# Solution: Run with flag
streamlit run streamlit_app.py --logger.level=debug
```

### Issue: Model accuracy is low
- Check dataset path: `merged_dataset/merged_dataset_cleaned.csv`
- Ensure hand is clearly visible in frame
- Test with different lighting conditions

## 📂 File Paths

```
Project Root/
├── training_model.ipynb       ← Run this to train
├── streamlit_app.py           ← Web UI
├── gesture_keyboard.py        ← Real-time control
├── hand_landmarker.task       ← Must be here (auto-downloads)
├── models/
│   ├── hand_gesture_model.h5  ← Saved model
│   ├── scaler.pkl             ← Feature scaler
│   └── label_encoder.pkl      ← Gesture labels
└── merged_dataset/
    └── merged_dataset_cleaned.csv  ← Training data
```

## 🎯 Quick Commands

```bash
# Train models
jupyter notebook training_model.ipynb

# Start web app
streamlit run streamlit_app.py

# Real-time gesture control
python gesture_keyboard.py

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 💡 Tips

1. **First Run**: Training takes 5-10 minutes on CPU. GPU is much faster.
2. **Webcam Issues**: Try different USB ports or cameras
3. **Better Accuracy**: Ensure good lighting and clear hand visibility
4. **Gesture Customization**: Edit keybinds in `gesture_keyboard.py` line ~50
5. **Save Custom Models**: Models auto-save after training

## 🆘 Need Help?

- Check console output for error messages
- Ensure all dependencies installed: `pip list | grep -E "tensorflow|mediapipe|opencv"`
- Verify dataset exists: `ls merged_dataset/`
- Test individual components separately

---

**All set!** Start with the Streamlit app to see it in action. 🚀
