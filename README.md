# 🛡️ Invicti Sign2Sound: Real-Time ASL Translation Engine

## 📖 Project Overview
Sign2Sound is a lightweight, real-time American Sign Language (ASL) translation system built by Team Invicti. Using advanced computer vision and custom Neural Networks, the system translates live webcam gestures into readable text and synthesized speech.

By utilizing Google MediaPipe for skeletal extraction and applying strict translation-invariant mathematics, this engine bypasses the heavy processing requirements of traditional Convolutional Neural Networks (CNNs). This allows it to run flawlessly in real-time on standard CPU hardware — no GPU needed.

> **Note:** The system operates in two modes:
> - **Letter Mode**: Recognizes 24 static ASL alphabet letters (excluding dynamic gestures like 'J' and 'Z').
> - **Word Mode**: Recognizes 24 ASL words using a 2.5-second temporal recording window.

---

## ✨ Key Features
- **Real-Time Translation:** Captures and classifies hand gestures at 30 FPS with < 1-second latency.
- **Translation Invariance:** Utilizes wrist-relative 3D coordinate math, meaning the AI recognizes the hand shape regardless of where it is positioned on the screen.
- **Temporal Sequence Tracking (Word Mode):** Tracks hand and upper-body pose movements over a 2.5-second window (30 evenly-sampled frames) to recognize full words and dynamic signs.
- **Audio Accessibility:** Integrates an offline Text-to-Speech (TTS) engine running on a background daemon thread, ensuring the system speaks translated words without freezing the live video feed.
- **Modern GUI:** Features a sleek, responsive dark-mode interface built with CustomTkinter, including a live confidence tracker and automatic sentence building.
- **Temporal Smoothing:** Implements a rolling stability buffer across consecutive frames to prevent screen flickering and guarantee confident predictions.

---

## 🧠 System Architecture & Data Flow

1. **Vision Pipeline:** OpenCV captures the live RGB feed and passes it to Google MediaPipe.
2. **Feature Extraction:** 
   - *Letter Mode*: MediaPipe identifies 21 3D hand landmarks (63 total x, y, z coordinates).
   - *Word Mode*: MediaPipe identifies both hands + 8 upper body pose landmarks (150 total coordinates per frame).
3. **Mathematical Normalization:** Coordinates are converted to "wrist-relative" values and normalized via `StandardScaler`.
4. **Classification:** 
   - *Letter Mode*: A custom 4-layer Feed-Forward Dense Neural Network (DNN).
   - *Word Mode*: A lightweight **Separable 1D-CNN** (21K parameters, Pi-safe, 34KB TFLite).
5. **Output:** Prediction shown on UI and spoken via `pyttsx3`.

---

## 🤟 Recognized Words (24 Classes)
| | | | |
|---|---|---|---|
| DRINK | EAT | EMERGENCY | HELLO |
| HELP | HOSPITAL | MEDICINE | MORE |
| NO | PAIN | PLEASE | SICK |
| THANK YOU | WANT | YES | doctor |
| me | meet | on | they |
| visit | we | you | your |

---

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | **99.62%** |
| Total Parameters | **21,434** (~83 KB) |
| TFLite Size | **34.2 KB** |
| Architecture | Separable 1D-CNN |
| Hardware Target | CPU-only (Pi 3B+ compatible) |

---

## 🛠️ Requirements
Python 3.11 or 3.12 required.

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

**Step 1: Clone the repository**
```bash
git clone https://github.com/J-a-co-b/SignToSound_Invicti.git
cd SignToSound_Invicti
```

```bash
pip install -r requirements.txt
```

**Step 3: Launch the Sign2Sound UI**
```bash
python gui_app.py
```
*(Ensure your webcam is connected before launching the app!)*
