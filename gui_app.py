import os
import platform as _platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLIB_SKIP_ARGB_VISUALS'] = '1'

import math
import threading
import queue
import cv2
import numpy as np
import json
import time
import platform
import multiprocessing
import pyttsx3
from collections import deque
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Activation,
    SeparableConv1D, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

try:
    import joblib as _joblib
except ImportError:
    _joblib = None


class _NumpyScaler:
    """Minimal StandardScaler replacement using only numpy."""
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (X - self.mean_) / self.scale_


APP_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_IDXS = [11, 12, 13, 14, 15, 16, 23, 24]
WORD_FRAMES = 30


# ==================================================
# 1. THE PERSISTENT TTS WORKER
# ==================================================
def tts_worker(conn):
    while True:
        try:
            text = conn.recv()
            if text == "STOP":
                break
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except EOFError:
            break
        except Exception as e:
            print(f"TTS Worker Error: {e}")


# ==================================================
# 2. UI APPLICATION CLASS (Standard Tkinter)
# ==================================================
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignToSound (Jetson Edition)")
        self.root.geometry("1024x600")
        self.root.configure(bg="#1e1e1e")

        self.bg_color = "#1e1e1e"
        self.panel_bg = "#2d2d2d"
        self.btn_bg = "#3d3d3d"
        self.btn_fg = "#ffffff"
        self.accent_color = "#6366f1"
        self.text_color = "#ffffff"
        self.sub_text_color = "#aaaaaa"

        self.displayed_word = ""
        self.target_word = ""
        self.typewriter_job = None

        # Multiprocessing TTS Pipe
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(target=tts_worker, args=(self.child_conn,), daemon=True)
        self.proc.start()

        # --- LOAD LETTER MODEL ---
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(63,)),
            BatchNormalization(momentum=0.99, epsilon=0.001),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(momentum=0.99, epsilon=0.001),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(24, activation='softmax'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        try:
            w_data = np.load('sign_language_model_weights.npz')
            self.model.set_weights([w_data[f"arr_{i}"] for i in range(len(w_data.files))])
        except Exception as e:
            print(f"Fallback/Error loading letter weights: {e}")
            try:
                self.model.load_weights('sign_language_model.weights.h5')
            except:
                pass

        self.actions = np.array(['A','B','C','D','E','F','G','H','I',
                                 'K','L','M','N','O','P','Q','R','S',
                                 'T','U','V','W','X','Y'])

        # --- LOAD WORD MODEL ---
        with open("word_label_map.json") as f:
            self.word_labels = json.load(f)
        self.word_list = [self.word_labels[str(i)] for i in range(len(self.word_labels))]
        n_classes = len(self.word_labels)

        inp = Input(shape=(WORD_FRAMES, 150), name="landmarks")
        x = SeparableConv1D(64, kernel_size=3, padding="same",
                            depthwise_regularizer=l2(1e-4),
                            pointwise_regularizer=l2(1e-4),
                            name="sep_conv1")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)
        x = SeparableConv1D(64, kernel_size=5, padding="same",
                            depthwise_regularizer=l2(1e-4),
                            pointwise_regularizer=l2(1e-4),
                            name="sep_conv2")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)
        x = SeparableConv1D(32, kernel_size=7, padding="same",
                            depthwise_regularizer=l2(1e-4),
                            pointwise_regularizer=l2(1e-4),
                            name="sep_conv3")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.20)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.25)(x)
        out = Dense(n_classes, activation="softmax", name="predictions")(x)
        self.word_model = Model(inp, out, name="SignToSound_Word")
        self.word_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        try:
            w_data = np.load('word_model_weights.npz')
            self.word_model.set_weights([w_data[f"arr_{i}"] for i in range(len(w_data.files))])
        except Exception as e:
            print(f"Fallback/Error loading word weights: {e}")
            try:
                self.word_model.load_weights("word_model.weights.h5")
            except:
                pass

        # --- LOAD SCALERS ---
        def _load_scaler(npz_name, pkl_name):
            npz_path = os.path.join(os.getcwd(), npz_name)
            if os.path.exists(npz_path):
                d = np.load(npz_path)
                return _NumpyScaler(d['mean'], d['scale'])
            pkl_path = os.path.join(os.getcwd(), pkl_name)
            if os.path.exists(pkl_path) and _joblib is not None:
                try:
                    return _joblib.load(pkl_path)
                except Exception:
                    pass
            return None

        self.scaler      = _load_scaler('scaler.npz', 'scaler.pkl')
        self.word_scaler = _load_scaler('word_scaler.npz', 'word_scaler.pkl')

        # --- LOAD MEDIAPIPE DETECTORS ---
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        hand_model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        pose_model_path = os.path.join(SCRIPT_DIR, "pose_landmarker.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # Variables
        self.prediction_buffer = deque(maxlen=8)
        self.current_stable_letter = ""
        self.stable_frames = 0
        self.CONFIDENCE_THRESHOLD = 0.80
        self.REQUIRED_FRAMES = 5
        self.letter_buffer = []
        self.word = ""
        self.last_seen_time = time.time()
        self.PAUSE_TIME = 1.5
        self._enhanced = False

        self.RECORD_DURATION   = 2.5
        self.word_raw_buffer   = []
        self.word_recording    = False
        self.word_record_start = 0.0
        self.word_cooldown_until = 0.0

        self.cap = self._open_camera()

        self._is_jetson = platform.machine() == 'aarch64'
        self._loop_delay   = 66 if self._is_jetson else 15
        self._infer_every  = 1
        self._frame_counter = 0

        self._mp_image_queue   = queue.Queue(maxsize=1)
        self._infer_result_queue = queue.Queue(maxsize=1)
        if self._is_jetson:
            threading.Thread(target=self._inference_thread, daemon=True).start()

        self.build_ui()
        self.update_video()
        threading.Thread(target=self._load_gec_model, daemon=True).start()

    def _open_camera(self):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            for i in range(1, 4):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    break
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def speak_word(self, text):
        if text.strip():
            try:
                self.parent_conn.send(text)
            except:
                pass
            self.animate_waveform(15)

    def animate_waveform(self, ticks):
        if ticks > 0:
            self.draw_waveform(1)
            self.root.after(100, lambda: self.animate_waveform(ticks-1))
        else:
            self.draw_waveform(0)

    def build_ui(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TProgressbar", thickness=10, background=self.accent_color, troughcolor=self.panel_bg)

        # Layout Frames
        self.top_frame = tk.Frame(self.root, bg=self.bg_color)
        self.top_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.bottom_frame = tk.Frame(self.root, bg=self.panel_bg, height=120, bd=2, relief="flat")
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.video_frame = tk.Frame(self.top_frame, bg="#000000", width=640, height=480)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = tk.Label(self.video_frame, text="Webcam Loading...", bg="#000000", fg="#ffffff")
        self.video_label.pack(fill="both", expand=True)

        self.info_frame = tk.Frame(self.top_frame, bg=self.panel_bg, width=340)
        self.info_frame.pack(side="right", fill="y", padx=0)

        # Info Frame Widgets
        tk.Label(self.info_frame, text="SignToSound", font=("Sans", 18, "bold"), bg=self.panel_bg, fg=self.text_color).pack(pady=(15, 5))

        self.conf_display = tk.Label(self.info_frame, text="Confidence: 0%", font=("Sans", 14), bg=self.panel_bg, fg="#22d3ee")
        self.conf_display.pack(pady=5)

        self.auto_mode_status = tk.Label(self.info_frame, text="Waiting for sign...", font=("Sans", 12), bg=self.panel_bg, fg=self.sub_text_color)
        self.auto_mode_status.pack(pady=5)

        self.word_progress = ttk.Progressbar(self.info_frame, style="TProgressbar", orient="horizontal", mode="determinate")
        self.word_progress.pack(fill="x", padx=20, pady=5)

        # Buttons
        btn_opts = {"bg": self.btn_bg, "fg": self.btn_fg, "activebackground": self.accent_color, "activeforeground": "#ffffff", "font": ("Sans", 11), "relief": "flat", "bd": 0, "pady": 6}

        row1 = tk.Frame(self.info_frame, bg=self.panel_bg)
        row1.pack(fill="x", padx=20, pady=(15, 5))
        tk.Button(row1, text="Space", command=self.add_space, **btn_opts).pack(fill="x")

        row2 = tk.Frame(self.info_frame, bg=self.panel_bg)
        row2.pack(fill="x", padx=20, pady=5)
        tk.Button(row2, text="Backspace", command=self.delete_last, **btn_opts).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(row2, text="Del Word", command=self.delete_word, **btn_opts).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(row2, text="Clear", command=self.clear_word, bg="#e5484d", fg="#ffffff", font=("Sans", 11), relief="flat", pady=6).pack(side="left", expand=True, fill="x", padx=(2, 0))

        self.enhance_btn = tk.Button(self.info_frame, text="Loading AI...", command=self.manual_speak, state="disabled", **btn_opts)
        self.enhance_btn.pack(fill="x", padx=20, pady=5)

        tk.Button(self.info_frame, text="Speak Out Loud", command=self.speak_out_loud, bg="#16a34a", fg="#ffffff", font=("Sans", 11, "bold"), relief="flat", pady=8).pack(fill="x", padx=20, pady=(5, 10))

        # Waveform Canvas
        self.wave_canvas = tk.Canvas(self.bottom_frame, width=120, height=40, bg=self.panel_bg, highlightthickness=0)
        self.wave_canvas.pack(side="right", padx=20, pady=20)
        self.draw_waveform(0)

        tk.Label(self.bottom_frame, text="Live Output:", font=("Sans", 12, "bold"), bg=self.panel_bg, fg=self.sub_text_color).pack(anchor="nw", padx=15, pady=(10, 0))
        
        self.word_display = tk.Label(self.bottom_frame, text="", font=("Sans", 26, "bold"), bg=self.panel_bg, fg="#e3b341", anchor="w", justify="left")
        self.word_display.pack(fill="x", padx=15, pady=(5, 10))

    def draw_waveform(self, intensity):
        self.wave_canvas.delete("all")
        import random
        for i in range(6):
            h = 4 if intensity == 0 else random.randint(10, 35)
            x = 10 + i * 18
            y = 35 - h
            self.wave_canvas.create_rectangle(x, y, x+12, 35, fill=self.accent_color, outline="")

    def set_target_word(self, new_word):
        self.target_word = new_word
        if len(self.displayed_word) > len(self.target_word):
            self.displayed_word = self.target_word
            self.word_display.config(text=self.displayed_word)
        if hasattr(self, 'typewriter_job') and self.typewriter_job:
            self.root.after_cancel(self.typewriter_job)
            self.typewriter_job = None
        self.update_typewriter()

    def update_typewriter(self):
        if len(self.displayed_word) < len(self.target_word):
            self.displayed_word = self.target_word[:len(self.displayed_word)+1]
            self.word_display.config(text=self.displayed_word)
            self.typewriter_job = self.root.after(40, self.update_typewriter)

    def add_space(self):
        if not self.word.endswith(" "):
            self.word += " "
            self.set_target_word(self.word)
            self.letter_buffer = []
            self._mark_content_changed()

    def delete_last(self):
        if self.word:
            self.word = self.word[:-1]
            self.set_target_word(self.word)
            self.letter_buffer = []

    def delete_word(self):
        text = self.word.rstrip(" ")
        if " " in text:
            self.word = text.rsplit(" ", 1)[0] + " "
        else:
            self.word = ""
        self.set_target_word(self.word)
        self.letter_buffer = []

    def clear_word(self):
        self.word = ""
        self.set_target_word("")
        self.letter_buffer = []

    def _mark_content_changed(self):
        if self._enhanced:
            self._enhanced = False
            if hasattr(self, 'enhance_btn'):
                self.enhance_btn.config(state="normal", bg=self.btn_bg)

    def speak_out_loud(self):
        text = self.word.strip()
        if text:
            self.speak_word(text)

    # GEC Model
    _GEC_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

    def _load_gec_model(self):
        self._gec_model    = None
        self._gec_tokenizer = None
        self._gec_ready    = False
        if not _TRANSFORMERS_AVAILABLE:
            print("[GEC] transformers not installed.")
            self.root.after(0, self._gec_mark_unavailable)
            return
        try:
            print("[GEC] Loading T5 grammar model...")
            tok   = T5Tokenizer.from_pretrained(self._GEC_MODEL_NAME)
            model = T5ForConditionalGeneration.from_pretrained(self._GEC_MODEL_NAME)
            model.eval()
            self._gec_tokenizer = tok
            self._gec_model     = model
            self._gec_ready     = True
            print("[GEC] T5 grammar model ready.")
            self.root.after(0, self._gec_mark_ready)
        except Exception as e:
            print(f"[GEC] Model load failed: {e}")
            self.root.after(0, self._gec_mark_unavailable)

    def _gec_mark_ready(self):
        if hasattr(self, 'enhance_btn'):
            self.enhance_btn.config(state="normal", text="Enhance Syntax", bg=self.accent_color)

    def _gec_mark_unavailable(self):
        if hasattr(self, 'enhance_btn'):
            self.enhance_btn.config(state="normal", text="Enhance (Basic)", bg=self.btn_bg)

    def _gec_correct(self, text):
        try:
            from grammar_engine import process_sentence
            raw_tokens = text.split()
            rule_based_result = process_sentence(raw_tokens)
            if not rule_based_result:
                rule_based_result = text.strip().lower()
        except Exception as e:
            rule_based_result = text.strip().lower()

        if self._gec_ready and self._gec_model is not None:
            try:
                import torch
                prompt = f"gec: {rule_based_result}"
                inputs = self._gec_tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                with torch.no_grad():
                    outputs = self._gec_model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
                result = self._gec_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return result[0].upper() + result[1:] if result else rule_based_result.capitalize()
            except Exception:
                pass
        return rule_based_result[0].upper() + rule_based_result[1:] if rule_based_result else ""

    def manual_speak(self):
        if not self.word.strip() or self._enhanced:
            return
        self.enhance_btn.config(state="disabled", text="Enhancing...")
        raw_text = self.word.strip()

        def _run():
            corrected = self._gec_correct(raw_text)
            def _apply():
                self.word = corrected + " "
                self.set_target_word(self.word)
                self._enhanced = True
                self.enhance_btn.config(state="disabled", text="Enhance Syntax")
            self.root.after(0, _apply)

        threading.Thread(target=_run, daemon=True).start()

    def _extract_word_features(self, mp_image):
        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)
        hand_detected = bool(hand_result.hand_landmarks)

        slots = {0: np.zeros(63, dtype=np.float32), 1: np.zeros(63, dtype=np.float32)}
        if hand_result.hand_landmarks:
            for hand, cat in zip(hand_result.hand_landmarks, hand_result.handedness):
                side = 0 if cat[0].category_name == "Left" else 1
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                pts -= pts[0]
                slots[side] = pts.flatten()
        hand_vec = np.concatenate([slots[0], slots[1]])

        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in POSE_IDXS], dtype=np.float32)
            anchor = (pts[0] + pts[1]) / 2.0
            pts -= anchor
            pose_vec = pts.flatten()
        else:
            pose_vec = np.zeros(24, dtype=np.float32)

        return np.concatenate([hand_vec, pose_vec]), hand_detected, hand_result

    def apply_video_effects(self, frame, hand_result):
        return frame

    def _inference_thread(self):
        while True:
            try:
                mp_image = self._mp_image_queue.get(timeout=1.0)
                result = self._process_frame(mp_image)
                try:
                    self._infer_result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._infer_result_queue.put(result)
            except queue.Empty:
                continue

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.video_label.config(text="[X] Camera not found!", image="")
            self.root.after(1000, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._is_jetson:
            try:
                self._mp_image_queue.put_nowait(mp_image)
            except queue.Full:
                pass
            try:
                hand_result = self._infer_result_queue.get_nowait()
                self._last_hand_result = hand_result
            except queue.Empty:
                hand_result = getattr(self, '_last_hand_result', None)
        else:
            self._frame_counter += 1
            hand_result = self._process_frame(mp_image)
            self._last_hand_result = hand_result

        frame = self.apply_video_effects(frame, hand_result)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb)
        v_width  = self.video_frame.winfo_width()
        v_height = self.video_frame.winfo_height()
        if v_width < 10 or v_height < 10:
            v_width, v_height = 640, 480

        resize_method = Image.BILINEAR if self._is_jetson else Image.LANCZOS
        img = img.resize((v_width, v_height), resize_method)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(text="", image=imgtk)

        self.root.after(self._loop_delay, self.update_video)

    def _process_frame(self, mp_image):
        now = time.time()
        features, hand_detected, hand_result = self._extract_word_features(mp_image)

        if not hand_detected:
            self.stable_frames = 0
            self.current_stable_letter = ""
            self.word_recording = False
            self.word_raw_buffer.clear()
            self.word_progress["value"] = 0
            self.conf_display.config(text="Confidence: 0%")
            self.auto_mode_status.config(text="Waiting for sign...")
            if len(self.letter_buffer) > 0 and (now - self.last_seen_time) > self.PAUSE_TIME:
                self.letter_buffer = []
            return hand_result

        letter_conf = 0.0
        letter_label = ""
        
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            wrist_x, wrist_y, wrist_z = hand[0].x, hand[0].y, hand[0].z
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            landmarks = np.array(landmarks, dtype=np.float32)
            if self.scaler is not None:
                landmarks = self.scaler.transform(landmarks.reshape(1, 63))
            else:
                max_val = np.max(np.abs(landmarks))
                if max_val > 0:
                    landmarks /= max_val
                landmarks = landmarks.reshape(1, 63)

            prediction = self.model.predict(landmarks, verbose=0)
            self.prediction_buffer.append(prediction)

            avg_pred = np.mean(self.prediction_buffer, axis=0)[0]
            letter_class_id = np.argmax(avg_pred)
            letter_conf = np.max(avg_pred)
            letter_label = self.actions[letter_class_id]

        if letter_conf >= self.CONFIDENCE_THRESHOLD:
            if letter_label == self.current_stable_letter:
                self.stable_frames += 1
            else:
                self.current_stable_letter = letter_label
                self.stable_frames = 1

            if self.stable_frames == self.REQUIRED_FRAMES:
                if len(self.letter_buffer) == 0 or letter_label != self.letter_buffer[-1]:
                    self.letter_buffer.append(letter_label)
                    self.word += letter_label
                    self.set_target_word(self.word)
                    self._mark_content_changed()
                self.last_seen_time = now
                
                self.word_recording = False
                self.word_raw_buffer.clear()
                self.word_progress["value"] = 0
                self.auto_mode_status.config(text=f"Letter: {letter_label}")
                self.conf_display.config(text=f"Confidence: {int(letter_conf * 100)}%")
                return hand_result
        else:
            self.stable_frames = 0
            
        if now < self.word_cooldown_until:
            self.auto_mode_status.config(text="Cooldown...")
            self.conf_display.config(text=f"Confidence: {int(letter_conf * 100)}%")
            return hand_result

        if not self.word_recording:
            self.word_recording = True
            self.word_record_start = now
            self.word_raw_buffer = []
            self.word_progress["value"] = 0
            
        elapsed = now - self.word_record_start
        progress = min(100.0, (elapsed / self.RECORD_DURATION) * 100)
        self.word_progress["value"] = progress
        
        self.word_raw_buffer.append(features)

        if elapsed < self.RECORD_DURATION:
            self.auto_mode_status.config(text=f"[REC] Word... {elapsed:.1f}s")
            self.conf_display.config(text=f"Confidence: {int(letter_conf * 100)}%")
        else:
            self.word_recording = False
            raw = self.word_raw_buffer
            self.word_raw_buffer = []
            
            n_collected = len(raw)
            if n_collected >= 5:
                idxs = np.linspace(0, n_collected - 1, WORD_FRAMES, dtype=int)
                seq  = np.array([raw[i] for i in idxs], dtype=np.float32)
                if self.word_scaler is not None:
                    flat = self.word_scaler.transform(seq.reshape(1, -1))
                    seq  = flat.reshape(1, WORD_FRAMES, 150)
                else:
                    seq = seq.reshape(1, WORD_FRAMES, 150)

                probs     = self.word_model.predict(seq, verbose=0)[0]
                class_id  = int(np.argmax(probs))
                word_confidence = float(probs[class_id])
                word_label = self.word_labels[str(class_id)]

                entropy     = -np.sum(probs * np.log(probs + 1e-9))
                max_entropy = np.log(len(probs))
                
                if entropy <= 0.75 * max_entropy and word_confidence > 0.80:
                    self.conf_display.config(text=f"Word Conf: {int(word_confidence * 100)}%")
                    self.auto_mode_status.config(text=f"[OK] {word_label}")
                    
                    self.word += word_label + " "
                    self.set_target_word(self.word)
                    self._mark_content_changed()

                    self.word_cooldown_until = now + 1.5
                    return hand_result

            self.auto_mode_status.config(text="Not recognised.")
            self.word_recording = True
            self.word_record_start = now
            self.word_raw_buffer = [features]
            self.word_progress["value"] = 0

        return hand_result

    def on_close(self):
        try:
            self.parent_conn.send("STOP")
        except:
            pass
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_root = tk.Tk()
    app = SignLanguageApp(app_root)
    app_root.protocol("WM_DELETE_WINDOW", app.on_close)
    app_root.mainloop()