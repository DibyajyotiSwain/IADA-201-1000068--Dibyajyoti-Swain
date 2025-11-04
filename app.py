# ----------------------------------------------------------
# DRIVER DROWSINESS & DISTRACTION DETECTION - STREAMLIT APP
# Final Version: Stable, Accurate, and Realistic
# ----------------------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from collections import deque
import time

st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="😴")

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/drowsiness_model.h5')

model = load_model()
classes = ['Closed', 'No_Yawn', 'Open', 'Yawn']

label_map = {
    'Closed': 'Eyes Closed',
    'No_Yawn': 'Alert',
    'Open': 'Eyes Open',
    'Yawn': 'Yawning'
}

# ----------------------------------------------------------
# Prediction Helper
# ----------------------------------------------------------
def predict_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img).resize((224, 224))
    arr = np.array(img_pil) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds)
    conf = np.max(preds)
    return classes[idx], conf

# ----------------------------------------------------------
# Streamlit Layout
# ----------------------------------------------------------
st.title("Driver Drowsiness & Distraction Detection")
st.write("AI system to monitor driver's alertness in real-time using facial cues.")

mode = st.radio("Choose Input Mode", ["📷 Live Camera (Mode A)", "🖼️ Image Upload (Mode B)"])

# ----------------------------------------------------------
# Mode B: Static Image Upload
# ----------------------------------------------------------
if mode == "🖼️ Image Upload (Mode B)":
    uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        label, conf = predict_frame(np.array(img))

        st.subheader(f"Prediction: {label_map[label]}")
        st.write(f"Confidence: {conf*100:.2f}%")

        if label == 'Yawn' and conf > 0.7:
            st.warning("⚠️ Drowsiness detected (Yawn)")
        elif label == 'Closed' and conf > 0.7:
            st.error("🚨 Eyes Closed — driver might be asleep!")
        else:
            st.success("✅ Driver appears alert and attentive.")

# ----------------------------------------------------------
# Mode A: Live Camera - Enhanced Stability
# ----------------------------------------------------------
else:
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    # For smoothing
    recent_preds = deque(maxlen=15)  # last 15 frames (~1.5s)
    last_label = "Alert"
    last_alert_time = 0

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠️ Camera not detected.")
            break

        # Prediction every frame
        label, conf = predict_frame(frame)

        # Filter weak predictions
        if conf > 0.7:
            recent_preds.append(label)

        # Determine dominant prediction in recent frames
        if len(recent_preds) >= 10:
            current_label = max(set(recent_preds), key=recent_preds.count)
        else:
            current_label = "No_Yawn"  # default safe

        # Only trigger alert if same state persists for >1.5 seconds
        current_time = time.time()
        if current_label != last_label:
            last_label = current_label
            last_alert_time = current_time

        stable_duration = current_time - last_alert_time

        display_label = "Alert"
        color = (0, 255, 0)  # Green

        if current_label == "Closed" and stable_duration > 1.5:
            display_label = "Eyes Closed"
            color = (0, 0, 255)
        elif current_label == "Yawn" and stable_duration > 1.5:
            display_label = "Yawning"
            color = (0, 255, 255)
        elif current_label == "No_Yawn" and stable_duration > 1.5:
            display_label = "Alert"
            color = (0, 255, 0)
        elif current_label == "Open" and stable_duration > 1.5:
            display_label = "Alert"
            color = (0, 255, 0)

        # Draw label nicely
        cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"{display_label}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    st.write("Camera stopped.")
