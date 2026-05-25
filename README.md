# SignToSound — Sign Language to Speech Translation System
### by Team Invicti

---

## What Is SignToSound?

SignToSound is an AI-powered system that recognises American Sign Language (ASL) gestures through a camera and converts them into spoken words in real time. It works at two levels:

- **Letter Level** — Recognises 24 static ASL letters (A–Y, excluding J and Z which require motion)
- **Word Level** — Recognises 15 common words/phrases (HELLO, HELP, THANK YOU, YES, NO, DRINK, EAT, PLEASE, WANT, MORE, PAIN, SICK, EMERGENCY, HOSPITAL, MEDICINE)

The system is designed as an **educational kit** for learning sign language, with plans to expand into a web platform accessible from any computer.

---

## How It Works

```
Camera captures hand gestures
        ↓
MediaPipe extracts hand & body landmarks (keypoints)
        ↓
AI model classifies the gesture
        ↓
Text-to-speech speaks the prediction out loud
```

**Letter Mode:** Analyses a single video frame → 63 hand landmarks → DNN model → predicted letter

**Word Mode:** Buffers 30 video frames → 150 landmarks per frame (hands + upper body pose) → 1D-CNN model → predicted word

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| AI Framework | TensorFlow / Keras |
| Hand Detection | MediaPipe Hand Landmarker |
| Pose Detection | MediaPipe Pose Landmarker |
| GUI (Desktop) | CustomTkinter |
| Text-to-Speech | pyttsx3 |
| Data Format | NumPy (.npy) |

---

## Project Structure

```
SignToSound_Invicti-main/
│
├── train.py                  ← Train the letter recognition DNN
├── train_words.py            ← Train the word recognition 1D-CNN
├── preprocess.py             ← Extract hand keypoints from images
├── preprocess_words.py       ← Extract hand+pose keypoints from videos
├── gui_app.py                ← Desktop GUI application
├── realtime_test.py          ← Standalone real-time testing script
├── convert_to_tflite.py      ← Convert models for Pi deployment
├── pi_app.py                 ← Raspberry Pi application (planned)
│
├── sign_language_model.keras ← Trained letter model
├── word_model.keras          ← Trained word model
├── scaler.pkl                ← Letter feature scaler
├── word_scaler.pkl           ← Word feature scaler
├── word_label_map.json       ← Word index → label mapping
├── hand_landmarker.task      ← MediaPipe hand model
├── pose_landmarker.task      ← MediaPipe pose model
│
├── My_Keypoint_Data/         ← Processed letter keypoint data (.npy)
├── My_Word_Data/             ← Processed word sequence data (.npy)
├── Labeled/                  ← Raw MP4 videos of word signs
│
├── PLAN.md                   ← Full implementation plan
├── PROJECT_STATUS.txt        ← Progress report (this companion file)
├── README.md                 ← This file
└── requirements.txt          ← Python dependencies
```

---

## How to Run (Desktop)

```bash
# 1. Create virtual environment
python -m venv env
source env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the desktop GUI
python gui_app.py
```

---

## Team Documents

| File | Purpose |
|------|---------|
| `README.md` | Project overview (this file) |
| `PLAN.md` | Detailed implementation plan for Pi kit + web platform |
| `PROJECT_STATUS.txt` | Current progress, achievements, and next steps |

---

*Team Invicti — May 2026*
