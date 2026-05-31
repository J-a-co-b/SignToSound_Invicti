# 🛡️ Invicti Sign2Sound: Real-Time ASL Translation Engine

## 📖 Project Overview
Sign2Sound is a lightweight, real-time American Sign Language (ASL) translation system built by Team Invicti. Using advanced computer vision and custom Neural Networks, the system translates live webcam gestures into readable text and synthesized speech.

By utilizing Google MediaPipe for skeletal extraction and applying strict translation-invariant mathematics, this engine bypasses the heavy processing requirements of traditional Convolutional Neural Networks (CNNs). This allows it to run flawlessly in real-time on **standard CPU hardware — no GPU needed**.

> **Note:** The system operates in two modes:
> - **Letter Mode**: Recognizes 24 static ASL alphabet letters (excluding dynamic gestures like 'J' and 'Z').
> - **Word Mode**: Recognizes 24 ASL words using a 2.5-second temporal recording window.

---

## ✨ Key Features
- **Real-Time Translation:** Captures and classifies hand gestures at 30 FPS with < 1-second latency.
- **Translation Invariance:** Wrist-relative 3D coordinate math — the AI recognizes shapes regardless of position on screen.
- **Temporal Sequence Tracking (Word Mode):** Tracks hand + upper-body pose over a 2.5-second window (30 evenly-sampled frames) to recognize full words and dynamic signs.
- **Audio Output:** Offline Text-to-Speech engine (`pyttsx3`) runs on a background thread — speaks predictions without freezing the video feed.
- **Modern GUI:** Dark-mode interface built with CustomTkinter, including a live confidence bar, progress timer, and auto sentence builder.
- **Cross-Platform:** Works on Windows, macOS, and Linux. No CUDA/GPU required.

---

## 🧠 System Architecture & Data Flow

1. **Vision Pipeline:** OpenCV captures the live webcam feed and passes frames to MediaPipe.
2. **Feature Extraction:**
   - *Letter Mode*: 21 3D hand landmarks → 63 wrist-relative coordinates per frame.
   - *Word Mode*: Both hands + 8 upper-body pose landmarks → 150 coordinates per frame, recorded over 2.5 seconds.
3. **Normalisation:** `StandardScaler` normalises features to the training distribution.
4. **Classification:**
   - *Letter Mode*: 4-layer Feed-Forward Dense Neural Network (DNN).
   - *Word Mode*: Lightweight **Separable 1D-CNN** (21K parameters, Pi-safe, 34 KB TFLite).
5. **Output:** Prediction displayed on UI and spoken aloud via TTS.

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
| Word Model Accuracy | **99.4% val accuracy** |
| Word Model Parameters | **21,434** (~83 KB) |
| TFLite Size (Pi) | **34.2 KB** |
| Architecture | Separable 1D-CNN |
| Letter Model | 4-layer DNN, ~60K params |
| Hardware Target | CPU-only (Raspberry Pi 3B+ compatible) |

Both models use a **weights-only loading strategy** — architecture is defined in code, weight values are loaded from `.weights.h5` files. This makes the app compatible with **any TF/Keras version** without serialisation errors.

---

## 🛠️ Requirements
**Python 3.11 or 3.12 required.**

```bash
pip install -r requirements.txt
```

> All dependencies (TensorFlow, MediaPipe, CustomTkinter, pyttsx3, etc.) are pinned in `requirements.txt`. No extra setup needed.

---

## 🚀 How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/J-a-co-b/SignToSound_Invicti.git
cd SignToSound_Invicti
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch the app**
```bash
python gui_app.py
```
> Make sure your webcam is connected before launching!

---

## 🎮 How to Use
- **Letter Mode** — Sign individual ASL letters. Each confirmed letter is appended to the sentence bar.
- **Word Mode** — Switch using the toggle at the top. Show your hands → the app records for 2.5 seconds → predicts and speaks the word automatically. A progress bar shows the recording window in real time.
- **Speak button** — Manually speaks the full sentence aloud at any time.
- **Auto-Speak switch** — Automatically speaks each word as it's added.

---

## 📁 Files Included
| File | Purpose |
|------|---------|
| `gui_app.py` | Main application |
| `sign_language_model.keras` | Letter model (full Keras) |
| `sign_language_model.weights.h5` | Letter model weights (cross-version safe) |
| `word_model.keras` | Word model (full Keras) |
| `word_model.weights.h5` | Word model weights (cross-version safe) |
| `word_model.tflite` | Quantised word model for Raspberry Pi |
| `scaler.pkl` | Letter feature scaler |
| `word_scaler.pkl` | Word feature scaler |
| `word_label_map.json` | Word class index → label mapping |
| `hand_landmarker.task` | MediaPipe hand detection model |
| `pose_landmarker.task` | MediaPipe pose detection model |
| `requirements.txt` | Python dependencies |
