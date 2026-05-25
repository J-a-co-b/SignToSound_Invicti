# SignToSound — Project Implementation Plan
> **Project:** Sign Language to Speech Translation System  
> **Team:** Invicti  
> **Version:** 1.0  
> **Date:** May 2026

---

## 1. Project Summary

SignToSound is an AI-powered system that recognises American Sign Language (ASL) hand gestures in real time and converts them into spoken audio. The system is built around a trained deep learning model and is designed for two deployment targets:

| Track | Description |
|-------|-------------|
| **A — Educational Hardware Kit** | A self-contained physical device built on a Raspberry Pi 3 B+ with a camera, small screen, and speaker |
| **B — Web Platform** | A browser-based application accessible from any laptop or desktop computer |

Both tracks share the same AI core — the models are already trained and validated.

---

## 2. Current State (What's Already Built)

| Component | Status | Details |
|-----------|--------|---------|
| Letter recognition model | ✅ Complete | 24 ASL letters (A–Y, excl. J/Z), DNN, ~100% accuracy |
| Word recognition model | ✅ Complete | 15 words (DRINK, EAT, EMERGENCY, HELP, HELLO, etc.), 1D-CNN, 100% accuracy |
| Training pipeline | ✅ Complete | `train.py`, `train_words.py` |
| Data preprocessing | ✅ Complete | `preprocess.py`, `preprocess_words.py` |
| Desktop GUI app | ✅ Complete | `gui_app.py` — CustomTkinter, webcam, TTS, trainer mode |
| Video dataset | ✅ Complete | 70 MP4-extracted sequences across 15 word classes |

---

## 3. Hardware — Track A (Educational Kit)

### 3.1 Bill of Materials

| Component | Purpose | Notes |
|-----------|---------|-------|
| Raspberry Pi 3 B+ | Main processor | Include heatsink + fan (mandatory) |
| MicroSD Card (≥32 GB, Class 10) | OS + software storage | |
| USB Webcam | Capture hand signs | Any UVC-compatible (e.g. Logitech C270) |
| 2.4" SPI TFT Display (ILI9341) | Show predictions + UI | 320×240 resolution |
| Speaker or 3.5mm Headphones | Audio output for TTS | Uses Pi's 3.5mm jack |
| 5V 2.5A Power Supply | Stable power | Unstable power causes crashes |
| GPIO Jumper Wires | TFT connection | |
| Project enclosure / case | Housing for the kit | 3D printed or off-the-shelf |

### 3.2 TFT Screen Wiring (GPIO)

```
TFT Pin     →   Raspberry Pi GPIO
---------       ------------------
VCC         →   3.3V   (Pin 1)
GND         →   GND    (Pin 6)
CS          →   GPIO 8 / CE0  (Pin 24)
RESET       →   GPIO 25       (Pin 22)
DC / RS     →   GPIO 24       (Pin 18)
MOSI (SDI)  →   GPIO 10 / MOSI (Pin 19)
SCK         →   GPIO 11 / SCLK (Pin 23)
LED         →   3.3V   (Pin 1)
```

---

## 4. Software Architecture

### 4.1 System Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        SHARED AI CORE                             │
│                                                                   │
│   Webcam Frame (video input)                                      │
│         │                                                         │
│         ▼                                                         │
│   MediaPipe Landmarker  ──  Extracts hand/pose keypoints          │
│         │                                                         │
│         ▼                                                         │
│   Feature Vector (63-dim for letters / 150-dim × 30 for words)   │
│         │                                                         │
│         ├──── Letter Mode → DNN Model → Predicted Letter          │
│         └──── Word Mode   → 1D-CNN   → Predicted Word            │
│                                  │                                │
│                            TTS Audio Output                       │
└───────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
 Track A: Pi App                    Track B: Web Platform
 (Pygame on TFT screen)             (React + FastAPI)
```

### 4.2 Track A — Pi Software Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Operating System | Raspberry Pi OS Lite (64-bit) | Minimal footprint |
| Display Driver | `fbtft` kernel module | Maps SPI TFT to `/dev/fb1` |
| UI Framework | Pygame (renders to framebuffer) | Lightweight, works without desktop |
| AI Inference | `tflite-runtime` | 10× lighter than full TensorFlow |
| Landmark Detection | MediaPipe Tasks API | Already in use on desktop |
| Text-to-Speech | `espeak` (via subprocess) | Near-zero latency, offline |

> ⚠️ **Important:** Do NOT install full TensorFlow on the Pi 3 B+. It exceeds the RAM limit. Use `tflite-runtime` instead.

### 4.3 Track B — Web Platform Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite |
| Styling | Vanilla CSS (dark theme, glassmorphism) |
| Backend API | Python FastAPI |
| Real-time comms | WebSocket (frame streaming) |
| AI inference | TFLite (via `tflite-runtime` or full TF) |
| TTS in browser | Web Speech API (built into modern browsers) |
| Deployment option 1 | Pi as local server (classroom/LAN use) |
| Deployment option 2 | Cloud VM — GCP/AWS/Render (public access) |

---

## 5. Key Technical Steps

### Step 1 — Model Conversion (Desktop)
Convert the existing `.keras` models into the lightweight `.tflite` format required by both the Pi and the web backend.

**Produces:**
- `letter_model.tflite`
- `word_model.tflite`
- `letter_scaler_mean.npy` / `letter_scaler_scale.npy`
- `word_scaler_mean.npy` / `word_scaler_scale.npy`

### Step 2 — Pi Hardware Setup
1. Flash Raspberry Pi OS Lite (64-bit) to SD card
2. Enable SPI via `raspi-config`
3. Wire TFT screen to GPIO (see Section 3.2)
4. Configure `fbtft` overlay in `/boot/config.txt`
5. Install dependencies: `tflite-runtime`, `mediapipe`, `pygame`, `opencv`, `espeak`
6. Transfer model files to Pi via SSH (`scp`)

### Step 3 — Pi Application (`pi_app.py`)
A lightweight standalone application that:
- Reads webcam at 320×240 resolution, targeting 10 FPS
- Runs hand/pose landmark detection via MediaPipe
- Performs TFLite inference for letters or words
- Renders live UI to the TFT screen via Pygame
- Speaks predictions via `espeak`
- Supports **Letter Mode** (per-frame, hold-to-confirm) and **Word Mode** (30-frame buffer)
- Auto-starts on boot via a `systemd` service

### Step 4 — Web Backend (`api/`)
A FastAPI Python server that exposes the same inference pipeline over HTTP and WebSocket:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/labels` | GET | List of supported letters and words |
| `/predict/letter` | POST | Single JPEG frame → letter prediction |
| `/predict/word` | POST | 30 JPEG frames → word prediction |
| `/ws/live` | WebSocket | Real-time streaming inference |

### Step 5 — Web Frontend (`web/`)
A React web application with:

| Page | Content |
|------|---------|
| **Landing** | Hero section, how it works, hardware kit info, CTA |
| **Practice** | Live webcam + real-time predictions, sentence builder, TTS button |
| **Learn** | Visual ASL alphabet + word sign reference guide |
| **About** | About the project and team |

---

## 6. Performance Expectations

| Metric | Pi 3 B+ | Cloud Server (web) |
|--------|---------|-------------------|
| Frame rate | ~10 FPS | ~20–30 FPS |
| Letter prediction latency | ~100–200ms | ~50ms |
| Word prediction (30 frames) | ~3–6 seconds | ~1–2 seconds |
| RAM usage | ~700–900 MB | Unrestricted |
| Works offline? | ✅ Yes | ❌ Needs internet |

> 💡 **Note:** If frame rate on the Pi 3 B+ is too slow for word recognition, the word model can be retrained on 10-frame sequences (instead of 30) to match the Pi's real-world speed. This is a quick fix in `train_words.py`.

---

## 7. Deployment Options (Web Platform)

| Option | Host | Cost | Best For |
|--------|------|------|----------|
| **Pi as local server** | Raspberry Pi (LAN) | Free | Classroom demo, no internet |
| **Free cloud tier** | Render / Railway | Free (limited) | MVP / prototype launch |
| **Cloud VM** | GCP / AWS e2-standard-2 | ~$10–15/mo | Production public platform |
| **Frontend hosting** | Vercel (static) | Free | React app deployment |

---

## 8. Project Timeline (6 Weeks)

```
Week 1 — Model Conversion & Prep
  □ Convert .keras models → .tflite format
  □ Export scaler arrays to .npy
  □ Verify TFLite accuracy matches Keras accuracy
  □ Prepare file bundle for Pi transfer

Week 2 — Pi Hardware Setup
  □ Flash Raspberry Pi OS Lite
  □ Wire TFT screen and verify display
  □ Install all software dependencies on Pi
  □ Transfer model files, verify MediaPipe runs

Week 3 — Pi Application
  □ Build Pygame UI (letter mode)
  □ Integrate TFLite letter inference
  □ Build word mode (30-frame buffer)
  □ Integrate espeak TTS
  □ Configure systemd auto-start

Week 4 — Web Backend
  □ Build FastAPI server (REST endpoints)
  □ Add WebSocket streaming endpoint
  □ Test all endpoints with Postman/curl
  □ Containerise with Docker (optional)

Week 5 — Web Frontend
  □ Scaffold React app with Vite
  □ Build webcam feed component
  □ Build WebSocket-connected prediction panel
  □ Build Landing, Learn, and About pages
  □ Polish UI (dark theme, animations)

Week 6 — Integration & Launch
  □ End-to-end testing (browser → server → prediction → TTS)
  □ Deploy backend to cloud (or Pi local)
  □ Deploy frontend to Vercel
  □ Write user documentation / README
  □ Record demo video
```

---

## 9. Open Decisions for the Team

> **Q1 — Web Backend Location:**  
> Should the API server run on the Pi (LAN only, free) or on a cloud VM (public access, ~$10/mo)?

> **Q2 — User Accounts:**  
> Should the web platform require login so users can track their learning progress, or remain completely anonymous?

> **Q3 — Pi Version:**  
> The plan is designed for Pi 3 B+. Upgrading to a **Pi 4 (2GB)** would give 2–3× better performance and make word recognition noticeably smoother. Is this feasible?

---

## 10. File Structure (End State)

```
SignToSound_Invicti-main/
│
├── ── Existing (trained models & data) ────────────────────
│   ├── sign_language_model.keras
│   ├── word_model.keras
│   ├── scaler.pkl / word_scaler.pkl
│   ├── word_label_map.json
│   ├── My_Keypoint_Data/
│   ├── My_Word_Data/
│   └── Labeled/
│
├── ── Track A (Pi Kit) ─────────────────────────────────────
│   ├── convert_to_tflite.py     ← run on desktop first
│   ├── pi_app.py                ← main Pi application
│   ├── letter_model.tflite      ← generated by conversion
│   ├── word_model.tflite        ← generated by conversion
│   └── *.npy                    ← scaler arrays
│
├── ── Track B — Backend ────────────────────────────────────
│   └── api/
│       ├── main.py              ← FastAPI server
│       ├── inference.py         ← TFLite + MediaPipe engine
│       ├── requirements.txt
│       └── models/              ← copy of TFLite files here
│
└── ── Track B — Frontend ───────────────────────────────────
    └── web/
        ├── src/
        │   ├── pages/           ← Landing, Practice, Learn, About
        │   ├── components/      ← WebcamFeed, PredictionCard, etc.
        │   └── hooks/           ← useWebSocket, useCamera
        └── package.json
```

---

*Document prepared by Invicti team — SignToSound v1.0*
