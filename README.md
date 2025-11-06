# IADA-201-1000068--Dibyajyoti-Swain
Machine Learning and Deep Learning Summative Assignment

🚗 Driver Drowsiness & Distraction Detection System (AI-Based)

**Live Web Application**
**Streamlit App:** https://iada-201-1000068--dibyajyoti-swain-sycn35ymb9p6sade6r23qm.streamlit.app/

## 🧠 Project Overview

This project aims to prevent road accidents caused by **driver fatigue and distraction** using an AI-powered real-time monitoring system.
The model detects early signs of drowsiness such as **eye closure**, **yawning**, and **distraction** using deep learning and computer vision.
It then issues **visual and audio alerts** through a Streamlit-based dashboard to help drivers stay alert and safe.

---
## 🎯 Objective
* Detect and classify facial states into:
  **Eyes Open**, **Eyes Closed**, **Yawning**, and **No Yawn (Alert)**.
* Reduce road accidents caused by fatigue or inattention.
* Develop a real-time detection system deployable on any device.
* Integrate with an easy-to-use dashboard for live monitoring.
---
## 🧩 Dataset Information
The dataset contains facial images categorized into four classes:

| Class Name | Description                      |
| ---------- | -------------------------------- |
| `closed`   | Eyes closed (drowsy/sleeping)    |
| `no_yawn`  | Alert or distracted (no yawning) |
| `open`     | Eyes open (attentive)            |
| `yawn`     | Yawning (drowsy)                 |

* Total images (full dataset): ~2,700
* Split ratio: 70% Train, 15% Validation, 15% Test
* Image size: 224 × 224 pixels
* Format: `.jpg`

### 📦 Full Dataset Access
The full dataset used for training and testing can be accessed here:
👉 [Click to Open Dataset on Google Drive](https://drive.google.com/drive/folders/1qr8fCRjJmBEjqg-xNDkEzIzVqq5GZ8R7)

## ⚙️ Data Preprocessing

* Image normalization using `ImageDataGenerator(rescale=1./255)`
* Image augmentation (rotation, brightness, zoom, flipping)
* Balanced distribution of images across all 4 classes
* Batched training and validation sets for efficient model learning
  
---
## 🤖 Model Architecture

| Layer             | Description                                                                  |
| ----------------- | ---------------------------------------------------------------------------- |
| **Base Model**    | MobileNetV2 (pretrained on ImageNet)                                         |
| **Added Layers**  | GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.3) → Dense(4, Softmax) |
| **Optimizer**     | Adam (learning rate = 0.0001)                                                |
| **Loss Function** | Categorical Crossentropy                                                     |
| **Batch Size**    | 32                                                                           |
| **Epochs**        | 25                                                                           |

---

## 📈 Model Evaluation

| Dataset    | Accuracy  | Loss |
| ---------- | --------- | ---- |
| Training   | 94.8%     | 0.22 |
| Validation | 91.2%     | 0.35 |
| Test       | **90.0%** | 0.41 |

### ✅ Final Test Accuracy **90.30%**

---
## 📊 Evaluation Metrics

| Metric        | Formula                                         | Result   |
| ------------- | ----------------------------------------------- | -------- |
| **Precision** | TP / (TP + FP)                                  | 0.91     |
| **Recall**    | TP / (TP + FN)                                  | 0.89     |
| **F1 Score**  | 2 × (Precision × Recall) / (Precision + Recall) | **0.90** |

---

## 💡 Project Insights & Impact

* Detects driver fatigue and distraction early to **prevent accidents**.
* Works efficiently in **real-time** on standard laptops.
* Can be integrated with **smart vehicle systems** or IoT dashboards.
* Lightweight model suitable for **edge deployment** (low-power devices).
* Demonstrates the practical use of **AI and Computer Vision** for social good.

---
## 🖥️ Application Workflow

1. **Model Training** — Performed using MobileNetV2 with transfer learning.
2. **Model Testing** — Evaluated on unseen facial images.
3. **App Development** — Built using Streamlit for live detection.
4. **Modes Available:**
   * 📷 **Live Camera Mode:** Real-time detection using webcam
   * 🖼️ **Image Upload Mode:** Upload a photo to test predictions
5. **Alerts Generated:**
   * Red: Eyes Closed
   * Yellow: Yawning
   * Green: Alert

---

## 🧮 Model Evaluation Summary

| Class   | Precision | Recall | F1 Score |
| ------- | --------- | ------ | -------- |
| Closed  | 0.90      | 0.93   | 0.91     |
| No_Yawn | 0.88      | 0.89   | 0.88     |
| Open    | 0.92      | 0.91   | 0.91     |
| Yawn    | 0.91      | 0.87   | 0.89     |

**Macro Avg F1:** 0.90
**Weighted Avg F1:** 0.90

---

## 🧾 Files Included in Repository

| File/Folder         | Description                                       |
| ------------------- | ------------------------------------------------- |
| `app.py`            | Streamlit web app for real-time detection         |
| `train_model.ipynb` | Model training notebook                           |
| `requirements.txt`  | List of dependencies                              |
| `models/`           | Contains trained model (`drowsiness_model.h5`) |
| `sample/`           | Contains screenshots of project results           |
| `README.md`         | Project documentation                             |

---

## 🏁 Conclusion

This project demonstrates how **AI and Deep Learning** can be applied to enhance road safety.
Through effective classification of facial states and real-time monitoring, it helps prevent driver fatigue-related accidents and serves as a strong example of **applied computer vision for human safety**.
