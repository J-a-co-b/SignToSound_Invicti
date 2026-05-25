# 🛡️ Invicti Sign2Sound: Real-Time ASL Translation Engine

## 📖 Project Overview
Sign2Sound is a lightweight, real-time American Sign Language (ASL) translation system built by Team Invicti. Using advanced computer vision and custom Neural Networks, the system translates live webcam gestures into readable text and synthesized speech.

By utilizing Google MediaPipe for skeletal extraction and applying strict translation-invariant mathematics, this engine bypasses the heavy processing requirements of traditional Convolutional Neural Networks (CNNs). This allows it to run flawlessly in real-time on standard CPU hardware.

> **Note:** The system operates in two modes:
> - **Letter Mode**: Recognizes 24 static ASL alphabet letters (excluding dynamic gestures like 'J' and 'Z').
> - **Word Mode**: Recognizes 15 highly-used ASL words (e.g., HELLO, HELP, EMERGENCY) using a 30-frame temporal buffer.

---

## ✨ Key Features
- **Real-Time Translation:** Captures and classifies hand gestures at 30 FPS with < 1-second latency.
- **Translation Invariance:** Utilizes wrist-relative 3D coordinate math, meaning the AI recognizes the hand shape regardless of where it is positioned on the screen.
- **Temporal Sequence Tracking (Word Mode):** Tracks hand and upper-body pose movements over 30 frames to recognize full words and dynamic signs.
- **Audio Accessibility:** Integrates an offline Text-to-Speech (TTS) engine running on a background daemon thread, ensuring the system speaks translated words without freezing the live video feed.
- **Modern GUI:** Features a sleek, responsive dark-mode interface built with CustomTkinter, including a live confidence tracker and automatic sentence building.
- **Temporal Smoothing:** Implements a rolling stability buffer across consecutive frames to prevent screen flickering and guarantee confident predictions.

---

## 🧠 System Architecture & Data Flow
Our pipeline completely isolates the geometry of the hand from environmental noise (like background clutter or lighting changes):

1. **Vision Pipeline:** OpenCV captures the live RGB feed and passes it to Google MediaPipe.
2. **Feature Extraction:** 
   - *Letter Mode*: MediaPipe identifies 21 3D hand landmarks (63 total x, y, z coordinates).
   - *Word Mode*: MediaPipe identifies both hands + 8 upper body pose landmarks (150 total coordinates per frame).
3. **Mathematical Normalization:** The coordinates are converted to "wrist-relative" values (subtracting the wrist/body anchor position) and normalized via `StandardScaler`.
4. **Classification:** 
   - *Letter Mode*: A custom 4-layer Feed-Forward Dense Neural Network (DNN) processes the 1D coordinate array.
   - *Word Mode*: A 1D Convolutional Neural Network (1D-CNN) processes the 30-frame sequence.
5. **Output:** The predicted character or word is bridged to the CustomTkinter UI via Pillow (PIL) and spoken via `pyttsx3`.

---

## 📊 Dataset & Augmentation
The models were trained on custom datasets of ASL spatial coordinates. To prevent overfitting and drastically improve real-world generalization, we applied Synthetic Jitter Augmentation.

By injecting Gaussian noise, applying random scaling, and mirroring into the coordinate arrays during training, we simulated natural human hand-shaking and different body sizes. This expanded our dataset by 400% and stabilized the model's real-world accuracy to ~100% on the test set.

---

## 🛠️ Requirements and Dependencies
This project was developed using Python 3.11/3.12. All necessary library versions (including OpenCV, MediaPipe, and TensorFlow) are strictly defined to ensure a stable build.

To install everything instantly, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project
*Note: This repository contains the optimized, production-ready inference engine. The training data and scripts have been excluded to keep the repository lightweight and runnable on any system.*

**Step 1: Clone the repository & enter the folder**
```bash
git clone https://github.com/J-a-co-b/SignToSound_Invicti.git
cd SignToSound_Invicti
```

**Step 2: Install the dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Launch the Sign2Sound UI**
```bash
python gui_app.py
```
*(Ensure your webcam is connected before launching the app!)*
