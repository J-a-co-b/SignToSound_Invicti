import cv2
import numpy as np
import os
import json
import time
import platform
import multiprocessing
import pyttsx3
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk
import joblib

# ==================================================
# Pose landmark indices for word-mode feature extraction
# ==================================================
POSE_IDXS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
WORD_FRAMES = 30  # frames to buffer before word prediction


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
# 2. UI APPLICATION CLASS
# ==================================================
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignToSound")
        self.root.geometry("1000x680")
        self.root.resizable(False, False)

        # --- MULTIPROCESSING TTS PIPE ---
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(target=tts_worker, args=(self.child_conn,), daemon=True)
        self.proc.start()

        # --- LOAD LETTER MODEL ---
        self.model = load_model("sign_language_model.keras")
        self.DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")
        self.actions = np.array(sorted([
            d for d in os.listdir(self.DATA_PATH)
            if os.path.isdir(os.path.join(self.DATA_PATH, d)) and len(d) == 1 and d.isalpha()
        ]))

        # --- LOAD WORD MODEL ---
        self.word_model = load_model("word_model.keras")
        with open("word_label_map.json") as f:
            self.word_labels = json.load(f)  # {"0": "DRINK", ...}
        self.word_list = [self.word_labels[str(i)] for i in range(len(self.word_labels))]

        # --- LOAD SCALERS ---
        scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        if self.scaler is None:
            print("⚠️  scaler.pkl not found – using per-sample max normalisation.")

        word_scaler_path = os.path.join(os.getcwd(), "word_scaler.pkl")
        self.word_scaler = joblib.load(word_scaler_path) if os.path.exists(word_scaler_path) else None
        if self.word_scaler is None:
            print("⚠️  word_scaler.pkl not found – word mode may be less accurate.")

        # --- LOAD MEDIAPIPE DETECTORS ---
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

        # Hand detector
        hand_model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # Pose detector (for word mode)
        pose_model_path = os.path.join(SCRIPT_DIR, "pose_landmarker.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # --- MODE ---
        self.mode_var = ctk.StringVar(value="LETTER")  # "LETTER" or "WORD"

        # --- LETTER MODE VARIABLES ---
        self.prediction_buffer = deque(maxlen=5)
        self.current_stable_letter = ""
        self.stable_frames = 0
        self.CONFIDENCE_THRESHOLD = 0.70
        self.REQUIRED_FRAMES = 5
        self.letter_buffer = []
        self.word = ""
        self.last_seen_time = time.time()
        self.PAUSE_TIME = 1.5
        self.auto_speak_var = ctk.BooleanVar(value=False)

        # --- WORD MODE VARIABLES ---
        # Timed recording: collect ALL frames for RECORD_DURATION seconds,
        # then evenly subsample to 30 — matching training's sample_frames() logic.
        self.RECORD_DURATION   = 2.5   # seconds (match typical sign duration)
        self.word_raw_buffer   = []    # stores ALL feature vecs during recording window
        self.word_recording    = False # True while actively recording
        self.word_record_start = 0.0   # time.time() when recording started
        self.word_cooldown_until = 0.0
        self.last_word_prediction = ""
        self.last_word_confidence = 0.0

        # --- CAMERA ---
        self.cap = self._open_camera()

        self.build_ui()
        self.update_video()

    # --------------------------------------------------
    def _open_camera(self):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            print("WARNING: Camera index 0 failed. Trying 1-3...")
            for i in range(1, 4):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    print(f"Camera found at index {i}")
                    break
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    # --------------------------------------------------
    def speak_word(self, text):
        if text.strip():
            try:
                self.parent_conn.send(text)
            except:
                pass

    # --------------------------------------------------
    def build_ui(self):
        # ── Video panel ──────────────────────────────
        self.video_frame = ctk.CTkFrame(self.root, width=660, height=500, corner_radius=10)
        self.video_frame.place(x=20, y=20)
        self.video_label = ctk.CTkLabel(self.video_frame, text="Webcam Loading...")
        self.video_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        # ── Right info panel ─────────────────────────
        self.info_frame = ctk.CTkFrame(self.root, width=280, height=500, corner_radius=10)
        self.info_frame.place(x=700, y=20)

        # ── Mode selector ────────────────────────────
        ctk.CTkLabel(self.info_frame, text="Mode",
                     font=("Arial", 14, "bold")).place(x=20, y=15)

        self.mode_seg = ctk.CTkSegmentedButton(
            self.info_frame, values=["LETTER", "WORD"],
            variable=self.mode_var,
            command=self._on_mode_change,
            font=("Arial", 14, "bold"),
            selected_color="#6C63FF",
            selected_hover_color="#5A52E0",
            width=240
        )
        self.mode_seg.place(x=20, y=45)

        # ── Prediction display ───────────────────────
        ctk.CTkLabel(self.info_frame, text="Detected",
                     font=("Arial", 16, "bold")).place(x=20, y=95)

        self.letter_display = ctk.CTkLabel(self.info_frame, text="-",
                                           font=("Arial", 72, "bold"), text_color="#00FFCC")
        self.letter_display.place(relx=0.5, y=175, anchor=ctk.CENTER)

        self.conf_display = ctk.CTkLabel(self.info_frame, text="Confidence: 0%",
                                         font=("Arial", 14))
        self.conf_display.place(x=20, y=230)

        self.stable_display = ctk.CTkLabel(self.info_frame, text="Stability: 0/5",
                                           font=("Arial", 14))
        self.stable_display.place(x=20, y=258)

        # ── Word mode status (hidden initially) ──────
        self.word_mode_status = ctk.CTkLabel(
            self.info_frame, text="",
            font=("Arial", 13), text_color="#BDC3C7"
        )
        self.word_mode_status.place(x=20, y=258)
        self.word_mode_status.place_forget()  # hidden in letter mode

        # ── Word mode progress bar ───────────────────
        self.word_progress = ctk.CTkProgressBar(
            self.info_frame, width=240, height=12,
            corner_radius=6, progress_color="#6C63FF"
        )
        self.word_progress.set(0)
        self.word_progress.place(x=20, y=290)
        self.word_progress.place_forget()  # hidden in letter mode

        # ── Action buttons ───────────────────────────
        ctk.CTkButton(self.info_frame, text="␣ Space",
                      command=self.add_space, fg_color="#3498DB").place(relx=0.5, y=330, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="⌫ Delete",
                      command=self.delete_last, fg_color="#F39C12").place(relx=0.5, y=370, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="🗑️ Clear",
                      command=self.clear_word, fg_color="#E74C3C").place(relx=0.5, y=410, anchor=ctk.CENTER)
        ctk.CTkButton(self.info_frame, text="🔊 Speak",
                      command=self.manual_speak, fg_color="#2ECC71").place(relx=0.5, y=450, anchor=ctk.CENTER)

        self.auto_speak_switch = ctk.CTkSwitch(self.info_frame, text="Auto-Speak",
                                               variable=self.auto_speak_var)
        self.auto_speak_switch.place(relx=0.2, y=480)

        # ── Sentence bar ─────────────────────────────
        self.bottom_frame = ctk.CTkFrame(self.root, width=960, height=60, corner_radius=10)
        self.bottom_frame.place(x=20, y=540)
        ctk.CTkLabel(self.bottom_frame, text="Sentence:",
                     font=("Arial", 18, "bold")).place(x=20, y=15)
        self.word_display = ctk.CTkLabel(self.bottom_frame, text="",
                                         font=("Arial", 28, "bold"), text_color="#FFD700")
        self.word_display.place(x=130, y=12)

        # ── Word mode info bar ───────────────────────
        self.word_info_frame = ctk.CTkFrame(self.root, width=960, height=50, corner_radius=10)
        self.word_info_frame.place(x=20, y=615)

        self.word_info_label = ctk.CTkLabel(
            self.word_info_frame,
            text="📝 LETTER MODE — Sign individual letters to spell words",
            font=("Arial", 14),
            text_color="#BDC3C7"
        )
        self.word_info_label.place(x=20, y=12)

    # --------------------------------------------------
    def _on_mode_change(self, new_mode):
        self.word_raw_buffer.clear()
        self.prediction_buffer.clear()
        self.stable_frames = 0
        self.current_stable_letter = ""

        if new_mode == "WORD":
            # Reset all recording state
            self.word_recording    = False
            self.word_raw_buffer   = []
            self.word_cooldown_until = 0
            # Switch UI
            self.letter_display.configure(text="—", text_color="#888888")
            self.conf_display.configure(text="Confidence: —")
            self.stable_display.place_forget()
            self.word_mode_status.place(x=20, y=258)
            self.word_mode_status.configure(text="Show your hands to start recording")
            self.word_progress.place(x=20, y=290)
            self.word_progress.set(0)
            self.word_info_label.configure(
                text=f"🤟 WORD MODE — Show hands, hold sign for {self.RECORD_DURATION:.0f}s, auto-predicts"
            )
        else:
            # Switch to letter mode UI
            self.letter_display.configure(text="-", text_color="#00FFCC")
            self.conf_display.configure(text="Confidence: 0%")
            self.stable_display.configure(text="Stability: 0/5")
            self.stable_display.place(x=20, y=258)
            self.word_mode_status.place_forget()
            self.word_progress.place_forget()
            self.word_info_label.configure(
                text="📝 LETTER MODE — Sign individual letters to spell words"
            )

    # --------------------------------------------------
    # Sentence controls
    def add_space(self):
        if not self.word.endswith(" "):
            if self.auto_speak_var.get() and self.word.strip():
                self.speak_word(self.word.split()[-1])
            self.word += " "
            self.word_display.configure(text=self.word)
            self.letter_buffer = []

    def delete_last(self):
        if self.word:
            self.word = self.word[:-1]
            self.word_display.configure(text=self.word)
            self.letter_buffer = []

    def clear_word(self):
        self.word = ""
        self.letter_buffer = []
        self.word_display.configure(text="")

    def manual_speak(self):
        self.speak_word(self.word)

    # --------------------------------------------------
    # Word mode: extract 150-dim feature vector from a frame
    # --------------------------------------------------
    def _extract_word_features(self, mp_image):
        """Extract 150-dim hand+pose feature vector. Returns (vec, hand_detected)."""
        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)

        hand_detected = bool(hand_result.hand_landmarks)

        # Hands: left (63) + right (63)
        slots = {0: np.zeros(63, dtype=np.float32),
                 1: np.zeros(63, dtype=np.float32)}
        if hand_result.hand_landmarks:
            for hand, cat in zip(hand_result.hand_landmarks, hand_result.handedness):
                side = 0 if cat[0].category_name == "Left" else 1
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                pts -= pts[0]  # wrist-relative
                slots[side] = pts.flatten()
        hand_vec = np.concatenate([slots[0], slots[1]])  # (126,)

        # Pose: 8 upper body landmarks (24)
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            pts = np.array([[lms[i].x, lms[i].y, lms[i].z]
                            for i in POSE_IDXS], dtype=np.float32)
            anchor = (pts[0] + pts[1]) / 2.0
            pts -= anchor
            pose_vec = pts.flatten()  # (24,)
        else:
            pose_vec = np.zeros(24, dtype=np.float32)

        return np.concatenate([hand_vec, pose_vec]), hand_detected  # (150,), bool

    # --------------------------------------------------
    def update_video(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.video_label.configure(
                text="❌ Camera not found!\nCheck connection or try a different USB port.\nRestart the app after reconnecting.",
                image=""
            )
            self.root.after(1000, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        current_mode = self.mode_var.get()

        if current_mode == "LETTER":
            self._process_letter_mode(mp_image)
        else:
            self._process_word_mode(mp_image)

        # Render frame
        img = Image.fromarray(rgb)
        img = img.resize((640, 480))
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
        self.video_label.configure(text="", image=imgtk)

        self.root.after(10, self.update_video)

    # --------------------------------------------------
    def _process_letter_mode(self, mp_image):
        """Single-frame letter recognition (existing logic)."""
        result = self.hand_detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist_x, wrist_y, wrist_z = hand[0].x, hand[0].y, hand[0].z
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            landmarks = np.array(landmarks, dtype=np.float32)

            # Use scaler if available, otherwise fall back to per-sample max
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
            class_id = np.argmax(avg_pred)
            confidence = np.max(avg_pred)
            letter = self.actions[class_id]

            self.conf_display.configure(text=f"Confidence: {int(confidence * 100)}%")

            if confidence > self.CONFIDENCE_THRESHOLD:
                if letter == self.current_stable_letter:
                    self.stable_frames += 1
                else:
                    self.current_stable_letter = letter
                    self.stable_frames = 1

                self.stable_display.configure(text=f"Stability: {self.stable_frames}/5")

                if self.stable_frames == self.REQUIRED_FRAMES:
                    self.letter_display.configure(text=letter)
                    if len(self.letter_buffer) == 0 or letter != self.letter_buffer[-1]:
                        self.letter_buffer.append(letter)
                        self.word += letter
                        self.word_display.configure(text=self.word)
                    self.last_seen_time = time.time()
            else:
                self.stable_frames = 0
        else:
            self.stable_frames = 0
            self.current_stable_letter = ""
            self.conf_display.configure(text="Confidence: 0%")
            self.stable_display.configure(text="Stability: 0/5")
            if len(self.letter_buffer) > 0 and (time.time() - self.last_seen_time) > self.PAUSE_TIME:
                self.letter_buffer = []

    # --------------------------------------------------
    def _process_word_mode(self, mp_image):
        """
        Timed recording window approach — matches training preprocessing exactly.

        Training (preprocess_words.py): reads ALL video frames, then evenly
        subsamples to 30 using np.linspace across the full sign duration.

        Real-time: collect ALL frames for RECORD_DURATION seconds (hands visible
        OR not — the sign may end before hands drop), then evenly subsample to 30.
        This ensures the temporal distribution matches training.
        """
        now = time.time()

        # ── Cooldown: show result, wait before next recording ──────────────
        if now < self.word_cooldown_until:
            return

        # ── Not recording yet: wait for hand to appear, then auto-start ────
        if not self.word_recording:
            features, hand_detected = self._extract_word_features(mp_image)
            if hand_detected:
                # Hand appeared — start recording
                self.word_recording    = True
                self.word_record_start = now
                self.word_raw_buffer   = [features]
                elapsed = 0.0
                self.word_progress.set(0)
                self.word_mode_status.configure(
                    text=f"🔴 Recording… 0.0 / {self.RECORD_DURATION:.1f}s")
                self.letter_display.configure(text="…", text_color="#FF9F43")
            else:
                self.word_progress.set(0)
                self.word_mode_status.configure(text="Show your hands to start recording")
                self.letter_display.configure(text="—", text_color="#888888")
                self.conf_display.configure(text="Confidence: —")
            return

        # ── Actively recording ─────────────────────────────────────────────
        elapsed = now - self.word_record_start
        progress = min(1.0, elapsed / self.RECORD_DURATION)
        self.word_progress.set(progress)
        remaining = max(0.0, self.RECORD_DURATION - elapsed)
        self.word_mode_status.configure(
            text=f"🔴 Recording… {elapsed:.1f} / {self.RECORD_DURATION:.1f}s")

        # Collect frame regardless of hand presence (sign may have just ended)
        features, _ = self._extract_word_features(mp_image)
        self.word_raw_buffer.append(features)

        # ── Recording window complete ──────────────────────────────────────
        if elapsed >= self.RECORD_DURATION:
            self.word_recording = False
            raw = self.word_raw_buffer
            self.word_raw_buffer = []

            n_collected = len(raw)
            if n_collected < 5:
                self.letter_display.configure(text="?", text_color="#888888")
                self.word_mode_status.configure(text="Too few frames — show hands and try again")
                self.word_cooldown_until = now + 1.0
                return

            # ── Evenly subsample to WORD_FRAMES — identical to training ──
            idxs = np.linspace(0, n_collected - 1, WORD_FRAMES, dtype=int)
            seq  = np.array([raw[i] for i in idxs], dtype=np.float32)  # (30, 150)

            # ── Normalise (same as training) ──────────────────────────────
            if self.word_scaler is not None:
                flat = self.word_scaler.transform(seq.reshape(1, -1))
                seq  = flat.reshape(1, WORD_FRAMES, 150)
            else:
                seq = seq.reshape(1, WORD_FRAMES, 150)

            # ── Predict ───────────────────────────────────────────────────
            probs     = self.word_model.predict(seq, verbose=0)[0]
            class_id  = int(np.argmax(probs))
            confidence = float(probs[class_id])
            word_label = self.word_labels[str(class_id)]

            # Entropy guard: reject if model is spread across all classes
            entropy     = -np.sum(probs * np.log(probs + 1e-9))
            max_entropy = np.log(len(probs))
            if entropy > 0.75 * max_entropy:
                self.letter_display.configure(text="?", text_color="#888888")
                self.conf_display.configure(text="Uncertain — try again")
                self.word_mode_status.configure(text="Not recognised — sign more clearly")
                self.word_cooldown_until = now + 1.5
                return

            self.last_word_prediction  = word_label
            self.last_word_confidence  = confidence

            col = "#00FFCC" if confidence > 0.80 else "#FF9F43"
            self.letter_display.configure(text=word_label, text_color=col)
            self.conf_display.configure(text=f"Confidence: {int(confidence * 100)}%")
            self.word_mode_status.configure(
                text=f"✅ {word_label}  ({n_collected} frames collected)" if confidence > 0.80
                     else f"⚠️  {word_label} — low confidence, try again"
            )
            self.word_progress.set(1.0)

            if confidence > 0.80:
                self.word += word_label + " "
                self.word_display.configure(text=self.word)
                if self.auto_speak_var.get():
                    self.speak_word(word_label)

            # Brief pause before next recording window opens
            self.word_cooldown_until = now + 1.5

    # --------------------------------------------------
    def on_close(self):
        try:
            self.parent_conn.send("STOP")
        except:
            pass
        self.cap.release()
        self.root.destroy()


# ==================================================
# 3. RUN
# ==================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app_root = ctk.CTk()
    app = SignLanguageApp(app_root)
    app_root.protocol("WM_DELETE_WINDOW", app.on_close)
    app_root.mainloop()