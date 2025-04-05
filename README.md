# 🤟 ASL Alphabet Recognition with CNN + Live Webcam Inference

This project demonstrates **American Sign Language (ASL) alphabet recognition** using a **Convolutional Neural Network (CNN)** trained on the [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet). It includes:

- **Image classification** of ASL hand signs (A–Z + special signs)
- **Real-time webcam-based prediction**
- **MediaPipe integration** for hand landmark detection (for visualization)

> Achieved **82% validation accuracy** after just **5 epochs** of training!

---

## 🚀 Features

- 📦 Loads the ASL Alphabet Dataset using KaggleHub
- 📊 Real-time image classification using TensorFlow/Keras CNN
- 🧠 Custom CNN model for accurate predictions
- 🖐️ MediaPipe-powered hand landmark detection for visualization
- 📸 JavaScript-based webcam capture inside Google Colab

---

## 📂 Dataset

- Dataset: [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)
- 29 classes: A–Z, plus "nothing", "space", and "del"
- Each letter has thousands of labeled images

---

## 🧠 Model Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')
])
Loss Function: SparseCategoricalCrossentropy

Optimizer: Adam

Trained for 5 epochs using ImageDataGenerator (with validation split 0.2)

🖥️ Live Inference (Webcam)
The notebook includes JavaScript code to capture an image from your webcam. The image is:

Resized to 64x64

Preprocessed (normalized)

Classified by the trained model

(Optional) Annotated with hand landmarks using MediaPipe


📈 Results
Validation Accuracy: 82% after 5 epochs

Predictions shown in real-time after webcam capture
"# Analyses-of-Machine-Learning-Techniques-for-Sign-Language-to-Text-conversion-for-Speech-Impaired" 
