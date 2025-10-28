# 💤 Driver Drowsiness Detection System

A real-time driver alertness monitoring system built using **Python**, **OpenCV**, and **TensorFlow/Keras**.  
This project detects driver drowsiness by monitoring eye closure patterns through a webcam feed.  
If prolonged eye closure is detected, an **alert sound** is triggered to prevent accidents.

---

## 🚀 Features
- 👁️ Real-time **face and eye detection** using Haar Cascades  
- 🧠 Deep learning model (`drowsiness.h5`) for **sleep detection**  
- 🧍 Live webcam monitoring of the driver’s face  
- ⚡ Lightweight and easy-to-run Jupyter Notebook version  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries Used:** OpenCV, TensorFlow/Keras, NumPy  
- **Model File:** `drowsiness.h5`  
- **Dataset Used:** [MRL Eye Dataset (Kaggle)](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)

---

## 📂 How It Works
1. The system captures live video feed from your webcam.  
2. Face and eyes are detected using Haar Cascade classifiers.  
3. The trained CNN model predicts whether eyes are open or closed.  
4. If eyes remain closed for a certain duration, the system sends an alert signal 

---

## 🧠 Model Training
The **Convolutional Neural Network (CNN)** model was trained using the  
[MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset),  
which contains thousands of labeled images of open and closed eyes captured under different lighting and orientations.

**Model Architecture (Sample):**
- **Input:** 24x24 grayscale eye images  
- **Conv2D (32 filters, 3x3, ReLU)**  
- **MaxPooling2D (2x2)**  
- **Conv2D (64 filters, 3x3, ReLU)**  
- **MaxPooling2D (2x2)**  
- **Flatten → Dense (128, ReLU)**  
- **Output:** Dense(1, Sigmoid) — predicts 0 (Open) or 1 (Closed)

The model was compiled with:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
