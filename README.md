
# Invicti Sign2Sound: Real-Time ASL Translation Engine

## Project Overview

Sign2Sound is a lightweight, real-time American Sign Language (ASL) translation system built by Team Invicti. Using advanced computer vision and custom Neural Networks, the system translates live webcam gestures into readable text and synthesized speech.

By utilizing Google MediaPipe for skeletal extraction and applying strict translation-invariant mathematics, this engine bypasses the heavy processing requirements of traditional Convolutional Neural Networks (CNNs). This allows it to run flawlessly in real-time on **standard CPU hardware — no GPU needed**.

> **Note:** The system operates in two modes:
> - **Letter Mode:** Recognizes 24 static ASL alphabet letters (excluding dynamic gestures like **J** and **Z**).
> - **Word Mode:** Recognizes 44 ASL words using a **2.5-second temporal recording window**.

---

## Key Features

- **Real-Time Translation** — Captures and classifies hand gestures at 30 FPS with less than 1-second latency.
- **Translation Invariance** — Wrist-relative 3D coordinate mathematics enables recognition regardless of hand position.
- **Temporal Sequence Tracking (Word Mode)** — Records 30 evenly sampled frames over 2.5 seconds using both hand and upper-body pose landmarks.
- **Offline Text-to-Speech** — Uses `pyttsx3` on a background thread without freezing the video feed.
- **Modern GUI** — Built with CustomTkinter featuring live confidence scores, recording progress, and an automatic sentence builder.
- **Cross-Platform** — Compatible with Windows, macOS, Linux, and Raspberry Pi. No CUDA or GPU required.

---

## System Architecture & Data Flow

```text
 Webcam
    │
    ▼
 OpenCV Video Capture
    │
    ▼
 Google MediaPipe
    │
    ├───────────────┐
    ▼               ▼
 Hand Landmarks   Pose Landmarks
    │               │
    └──────┬────────┘
           ▼
 Feature Extraction
           │
           ▼
 StandardScaler
           │
           ▼
  Neural Network
 (Letter / Word)
           │
           ▼
 Prediction
           │
     ┌─────┴─────┐
     ▼           ▼
 GUI Display   Text-to-Speech
```

### Processing Pipeline

1. **Vision Pipeline** — OpenCV captures the webcam feed and forwards frames to MediaPipe.
2. **Feature Extraction**
   - **Letter Mode:** 21 hand landmarks → 63 wrist-relative coordinates.
   - **Word Mode:** Both hands + 8 pose landmarks → 150 coordinates per frame over 30 frames.
3. **Normalization** — Features are normalized using `StandardScaler`.
4. **Classification**
   - **Letter Mode:** Four-layer Dense Neural Network.
   - **Word Mode:** Lightweight Separable 1D-CNN optimized for Raspberry Pi.
5. **Output** — Prediction is displayed in the GUI and spoken using offline TTS.

---

## Recognized Classes

### Letter Mode

24 static ASL alphabet letters

```text
A B C D E F G H I K L M
N O P Q R S T U V W X Y
```

*(Dynamic letters J and Z are intentionally excluded.)*

---

### Word Mode (44 Classes)

| | | | |
|---|---|---|---|
| DRINK | EAT | EMERGENCY | HELLO |
| HELP | HOSPITAL | MEDICINE | MORE |
| NO | PAIN | PLEASE | SICK |
| THANK YOU | WANT | YES | college |
| doctor | me | meet | on |
| parents | satisfy | their | them |
| then | they | visit | wait |
| war | way | we | wear |
| week | wheelchair | where | which |
| who | why | without | witness |
| wow | you | your | yourself |

---

## Model Performance

| Metric | Value |
|---------|------:|
| Word Model Accuracy | **97.09%** |
| Word Model Parameters | **21,434** |
| TFLite Model Size | **35.8 KB** |
| Architecture | **Separable 1D-CNN** |
| Letter Model | **4-Layer Dense Neural Network** |
| Hardware Target | **CPU Only (Raspberry Pi 3B+ Compatible)** |

Both models follow a **weights-only loading strategy**. Network architecture is defined directly in code while trained parameters are loaded from `.weights.h5` files, ensuring compatibility across different TensorFlow/Keras versions.

---

## Requirements

- Python **3.11** or **3.12**
- Webcam
- Microphone/Speakers (optional for TTS)

Install dependencies:

```bash
pip install -r requirements.txt
```

All required packages including TensorFlow, MediaPipe, OpenCV, CustomTkinter, and pyttsx3 are listed in `requirements.txt`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/J-a-co-b/SignToSound_Invicti.git
cd SignToSound_Invicti
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python gui_app.py
```

> Ensure your webcam is connected before launching the application.

---

## Usage

### Letter Mode

- Perform one ASL letter.
- The predicted character is added to the sentence builder.
- Confidence score updates live.

### Word Mode

- Switch to **Word Mode**.
- Hold the sign for approximately **2.5 seconds**.
- The application records 30 frames.
- The predicted word is automatically displayed and spoken.

### Speech Controls

- **Speak Button** — Reads the current sentence aloud.
- **Auto Speak** — Automatically speaks every accepted prediction.

---

## Repository Structure

| File | Description |
|------|-------------|
| `gui_app.py` | Main application |
| `train.py` | Letter model training |
| `train_words.py` | Word model training |
| `grammar_engine.py` | Sentence correction engine |
| `sign_language_model.keras` | Letter model |
| `sign_language_model.weights.h5` | Letter model weights |
| `sign_language_model.tflite` | Letter TFLite model |
| `word_model.keras` | Word model |
| `word_model.weights.h5` | Word model weights |
| `word_model.tflite` | Raspberry Pi optimized word model |
| `hand_landmarker.task` | MediaPipe hand detector |
| `pose_landmarker.task` | MediaPipe pose detector |
| `scaler.pkl` | Letter feature scaler |
| `word_scaler.pkl` | Word feature scaler |
| `label_map.json` | Letter labels |
| `word_label_map.json` | Word labels |
| `requirements.txt` | Python dependencies |

---

## Technology Stack

- Python
- TensorFlow / Keras
- TensorFlow Lite
- Google MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- CustomTkinter
- pyttsx3

---

## Team

**Team Invicti**

Developed as a real-time assistive communication system for American Sign Language translation.

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgements

- Google MediaPipe
- TensorFlow
- OpenCV
- Scikit-learn
- CustomTkinter
- Python Community
````
