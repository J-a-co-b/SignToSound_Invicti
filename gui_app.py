import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TF/CUDA verbose logs
os.environ['XLIB_SKIP_ARGB_VISUALS'] = '1' # Fix X11 BadLength RenderAddGlyphs error
import math
import threading
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
import cv2
import numpy as np
import os
import json
import time
import platform
import multiprocessing
import pyttsx3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Activation,
    SeparableConv1D, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk
# joblib kept for fallback only
try:
    import joblib as _joblib
except ImportError:
    _joblib = None


class _NumpyScaler:
    """Minimal StandardScaler replacement using only numpy — no sklearn needed."""
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (X - self.mean_) / self.scale_

APP_DIR = os.path.dirname(os.path.abspath(__file__))

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
        self.root.geometry("1100x720")
        self.root.resizable(True, True)
        try:
            self.root.state("zoomed")
        except:
            pass
        
        ctk.set_appearance_mode("Dark")
        # ── Palette: Deep Indigo ──
        self.bg_color = "#0A0E1A"
        self.card_color = "#131829"
        self.card_alt = "#1D2440"
        self.accent_primary = "#6366F1"     # indigo
        self.accent_secondary = "#22D3EE"   # cyan
        self.text_primary = "#F1F5F9"
        self.text_secondary = "#7C85A3"
        self.root.configure(fg_color=self.bg_color)

        self.displayed_word = ""
        self.target_word = ""
        self.typewriter_job = None

        # --- MULTIPROCESSING TTS PIPE ---
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

        # --- AUTO MODE VARIABLES ---
        self.emitted_letters = []

        # --- LETTER MODE VARIABLES ---
        self.prediction_buffer = deque(maxlen=5)
        self.current_stable_letter = ""
        self.stable_frames = 0
        self.CONFIDENCE_THRESHOLD = 0.80
        self.REQUIRED_FRAMES = 5
        self.letter_buffer = []
        self.word = ""
        self.last_seen_time = time.time()
        self.PAUSE_TIME = 1.5
        self._enhanced = False   # tracks whether current text has already been enhanced

        # --- WORD MODE VARIABLES ---
        self.RECORD_DURATION   = 2.5
        self.word_raw_buffer   = []
        self.word_recording    = False
        self.word_record_start = 0.0
        self.word_cooldown_until = 0.0
        self.last_word_prediction = ""
        self.last_word_confidence = 0.0

        # --- CAMERA ---
        self.cap = self._open_camera()

        self.build_ui()
        self.update_video()
        # Load GEC model in the background so the UI opens instantly
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
        self.f_title = ("TkDefaultFont", 24, "bold")
        self.f_header = ("TkDefaultFont", 16, "bold")
        self.f_body = ("TkDefaultFont", 14)
        self.f_large = ("TkDefaultFont", 48, "bold")
        
        # Size the video column to maintain 4:3 aspect ratio based on available window height.
        # This keeps the camera view natural (no zoom) and lets the right panel expand freely.
        self.root.update_idletasks()
        win_h = self.root.winfo_height()
        win_w = self.root.winfo_width()
        # Vertical overhead: title bar (~30) + top pady (20) + gap (10) + bottom bar (180) + bottom paddings (30)
        video_panel_h = max(300, win_h - 270)
        # Width to maintain 4:3, plus frame's own padding
        video_col_w = int(video_panel_h * 4 / 3) + 30
        # Never take more than 65% of window width
        video_col_w = min(video_col_w, int(win_w * 0.65))
        self.root.grid_columnconfigure(0, weight=0, minsize=video_col_w)
        self.root.grid_columnconfigure(1, weight=1)   # right panel expands
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0, minsize=180)
        
        # ── Video panel ──
        self.video_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color="#000000")
        self.video_frame.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Webcam Loading...", font=self.f_body, fg_color="transparent")
        self.video_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        # ── Info panel — all children packed, nothing ever clips ──
        self.info_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=self.card_color)
        self.info_frame.grid(row=0, column=1, padx=(10, 20), pady=(20, 10), sticky="nsew")



        # ── Confidence ring ──
        self.ring_canvas = ctk.CTkCanvas(
            self.info_frame, width=150, height=150,
            bg=self.card_color, highlightthickness=0
        )
        self.ring_canvas.pack(pady=(14, 0))

        # Static background track
        self.ring_canvas.create_oval(10, 10, 140, 140, outline=self.card_alt, width=12)

        self._ring_cx, self._ring_cy, self._ring_r = 75, 75, 65
        self._ring_steps = 200
        self._ring_grad_start = (99, 102, 241)
        self._ring_grad_end   = (34, 211, 238)
        self.ring_segments = []

        for i in range(self._ring_steps):
            factor = i / (self._ring_steps - 1)
            r = int(self._ring_grad_start[0] + (self._ring_grad_end[0] - self._ring_grad_start[0]) * factor)
            g = int(self._ring_grad_start[1] + (self._ring_grad_end[1] - self._ring_grad_start[1]) * factor)
            b = int(self._ring_grad_start[2] + (self._ring_grad_end[2] - self._ring_grad_start[2]) * factor)
            color = f"#{r:02x}{g:02x}{b:02x}"
            seg = self.ring_canvas.create_line(0, 0, 0, 0, fill=color, width=12,
                                                capstyle="round", state="hidden")
            self.ring_segments.append(seg)

        self.draw_confidence_ring(0)

        # Percentage label embedded inside the ring canvas (no clipping possible)
        self.ring_value_label = ctk.CTkLabel(
            self.ring_canvas, text="0%", font=("TkDefaultFont", 26, "bold"),
            text_color=self.accent_secondary, fg_color=self.card_color
        )
        self.ring_canvas.create_window(75, 75, window=self.ring_value_label)

        # Hidden letter display — not shown, kept for internal logic
        self.letter_display = ctk.CTkLabel(self.info_frame, text="", font=self.f_large)

        # Confidence label
        self.conf_display = ctk.CTkLabel(
            self.info_frame, text="Confidence: 0%",
            font=self.f_body, text_color=self.accent_secondary
        )
        self.conf_display.pack(pady=(10, 0))

        self.status_area = ctk.CTkFrame(self.info_frame, fg_color="transparent", height=30)
        self.status_area.pack(pady=(4, 0), padx=20, fill="x")

        self.auto_mode_status = ctk.CTkLabel(
            self.status_area, text="Waiting for sign...", font=self.f_body, text_color=self.text_secondary
        )
        self.auto_mode_status.pack()

        self.word_progress = ctk.CTkProgressBar(
            self.info_frame, height=8, progress_color=self.accent_primary
        )
        self.word_progress.pack(pady=(4, 0), padx=20, fill="x")
        self.word_progress.set(0)

        # ── Button row 1a: Space ──
        btn_space = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_space.pack(pady=(18, 0), padx=20, fill="x")
        ctk.CTkButton(btn_space, text="\u2423 Space", font=self.f_body,
                      command=self.add_space, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x")

        # ── Button row 1b: Backspace / Delete Word / Clear ──
        btn_row1 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row1.pack(pady=(6, 0), padx=20, fill="x")
        ctk.CTkButton(btn_row1, text="\u232b Backspace", font=self.f_body,
                      command=self.delete_last, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(btn_row1, text="\u2715 Del Word", font=self.f_body,
                      command=self.delete_word, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x", padx=4)
        ctk.CTkButton(btn_row1, text="\U0001f5d1\ufe0f Clear", font=self.f_body,
                      command=self.clear_word, fg_color="#E5484D", width=0
                      ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # ── Button row 2: Enhance (starts disabled until T5 loads) ──
        btn_row2 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row2.pack(pady=(8, 0), padx=20, fill="x")
        self.enhance_btn = ctk.CTkButton(
            btn_row2, text="⏳ Loading AI…", font=self.f_body,
            command=self.manual_speak, fg_color=self.card_alt,
            state="disabled", width=0
        )
        self.enhance_btn.pack(side="left", expand=True, fill="x")

        # ── Button row 3: Speak Out Loud ──
        btn_row3 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row3.pack(pady=(8, 0), padx=20, fill="x")
        ctk.CTkButton(
            btn_row3, text="\U0001f50a Speak Out Loud", font=self.f_body,
            command=self.speak_out_loud,
            fg_color="#16A34A", hover_color="#15803D", width=0
        ).pack(side="left", expand=True, fill="x")

        # ── Sentence / bottom bar ──
        self.bottom_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=self.card_color)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")
        
        self.wave_canvas = ctk.CTkCanvas(
            self.bottom_frame, width=100, height=40,
            bg=self.card_color, highlightthickness=0
        )
        self.wave_canvas.place(relx=1.0, x=-120, y=20)
        self.draw_waveform(0)

        ctk.CTkLabel(self.bottom_frame, text="Live Output:", font=self.f_header).place(x=20, y=20)
        self.word_display = ctk.CTkLabel(
            self.bottom_frame, text="",
            font=("TkDefaultFont", 32, "bold"), text_color="#E3B341", justify="left"
        )
        self.word_display.place(x=20, y=60)
        
        self.word_info_label = ctk.CTkLabel(
            self.bottom_frame,
            text="\U0001f916 AUTO MODE \u2014 Sign letters or words directly, the AI will auto-detect",
            font=self.f_body, text_color=self.text_secondary
        )
        self.word_info_label.place(x=20, rely=1.0, y=-40)
        
        def update_wraplength(event):
            self.word_display.configure(wraplength=event.width - 40)
        self.bottom_frame.bind("<Configure>", update_wraplength)

    def draw_confidence_ring(self, percentage):
        """Draws a smooth gradient ring sweeping clockwise from the top (12 o'clock).
        Turns green when percentage >= 80, otherwise indigo-to-cyan gradient."""
        if not hasattr(self, 'ring_segments'):
            return
        percentage = max(0, min(100, percentage))
        visible_steps = round((percentage / 100) * self._ring_steps)
        high_conf = percentage >= 80

        cx, cy, r = self._ring_cx, self._ring_cy, self._ring_r
        for i, seg in enumerate(self.ring_segments):
            if i >= visible_steps:
                self.ring_canvas.itemconfig(seg, state="hidden")
                continue
            frac = i / (self._ring_steps - 1)
            if high_conf:
                # Solid green gradient: dark green → bright green
                g_val = int(160 + 74 * frac)   # 160 → 234
                color = f"#22{g_val:02x}5E".replace("5E", f"{int(60 + 38*frac):02x}")
                # Simpler: blend #16A34A → #4ADE80
                r_c = int(0x16 + (0x4A - 0x16) * frac)
                g_c = int(0xA3 + (0xDE - 0xA3) * frac)
                b_c = int(0x4A + (0x80 - 0x4A) * frac)
                color = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
            else:
                # Default indigo → cyan gradient
                r_c = int(self._ring_grad_start[0] + (self._ring_grad_end[0] - self._ring_grad_start[0]) * frac)
                g_c = int(self._ring_grad_start[1] + (self._ring_grad_end[1] - self._ring_grad_start[1]) * frac)
                b_c = int(self._ring_grad_start[2] + (self._ring_grad_end[2] - self._ring_grad_start[2]) * frac)
                color = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
            half_step_angle = (360 / self._ring_steps) * 0.65
            a1 = math.radians(-90 + frac * 360 - half_step_angle)
            a2 = math.radians(-90 + frac * 360 + half_step_angle)
            x1 = cx + r * math.cos(a1)
            y1 = cy + r * math.sin(a1)
            x2 = cx + r * math.cos(a2)
            y2 = cy + r * math.sin(a2)
            self.ring_canvas.coords(seg, x1, y1, x2, y2)
            self.ring_canvas.itemconfig(seg, state="normal", fill=color)

        label_color = "#4ADE80" if high_conf else self.accent_secondary
        if hasattr(self, 'ring_value_label'):
            self.ring_value_label.configure(
                text=f"{int(percentage)}%",
                text_color=label_color
            )
        if hasattr(self, 'conf_display'):
            self.conf_display.configure(text_color=label_color)

    def draw_waveform(self, intensity):
        self.wave_canvas.delete("all")
        import random
        for i in range(5):
            h = 4 if intensity == 0 else random.randint(10, 35)
            x = 10 + i * 18
            y = 35 - h
            self.wave_canvas.create_rectangle(x, y, x+10, 35, fill=self.accent_primary, outline="")

    def set_target_word(self, new_word):
        self.target_word = new_word
        if len(self.displayed_word) > len(self.target_word):
            self.displayed_word = self.target_word
            self.word_display.configure(text=self.displayed_word)
        if hasattr(self, 'typewriter_job') and self.typewriter_job:
            self.root.after_cancel(self.typewriter_job)
            self.typewriter_job = None
        self.update_typewriter()

    def update_typewriter(self):
        if len(self.displayed_word) < len(self.target_word):
            self.displayed_word = self.target_word[:len(self.displayed_word)+1]
            self.word_display.configure(text=self.displayed_word)
            self.typewriter_job = self.root.after(40, self.update_typewriter)

    # Removed mode change logic since we are using unified Auto Mode

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
        """Delete the last word (or trailing space if cursor is after a space)."""
        text = self.word.rstrip(" ")
        if " " in text:
            # Trim back to the end of the previous word, keeping one trailing space
            self.word = text.rsplit(" ", 1)[0] + " "
        else:
            # Only one word left — clear it entirely
            self.word = ""
        self.set_target_word(self.word)
        self.letter_buffer = []

    def clear_word(self):
        self.word = ""
        self.set_target_word("")
        self.letter_buffer = []

    def _mark_content_changed(self):
        """Re-enable Enhance Syntax whenever new content is added."""
        if self._enhanced:
            self._enhanced = False
            if hasattr(self, 'enhance_btn'):
                self.enhance_btn.configure(state="normal", fg_color=self.accent_primary)

    def speak_out_loud(self):
        """Speak whatever is currently in the output box."""
        text = self.word.strip()
        if text:
            self.speak_word(text)

    # --------------------------------------------------
    # T5 Grammar Error Correction
    # --------------------------------------------------
    _GEC_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

    def _load_gec_model(self):
        """Load T5-small GEC model in background. Runs on CPU for Jetson Nano compatibility."""
        self._gec_model    = None
        self._gec_tokenizer = None
        self._gec_ready    = False
        if not _TRANSFORMERS_AVAILABLE:
            print("[GEC] transformers not installed — Enhance Syntax will use basic cleanup.")
            self.root.after(0, self._gec_mark_unavailable)
            return
        try:
            print("[GEC] Loading T5 grammar model…")
            tok   = T5Tokenizer.from_pretrained(self._GEC_MODEL_NAME)
            model = T5ForConditionalGeneration.from_pretrained(self._GEC_MODEL_NAME)
            model.eval()  # inference-only mode, saves memory
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
            self.enhance_btn.configure(
                state="normal",
                fg_color=self.accent_primary,
                text="\U0001fa84 Enhance Syntax"
            )

    def _gec_mark_unavailable(self):
        if hasattr(self, 'enhance_btn'):
            self.enhance_btn.configure(
                text="\U0001fa84 Enhance (basic)",
                state="normal",
                fg_color=self.card_alt
            )

    def _gec_correct(self, text):
        """Run Hybrid AI: local grammar_engine rules first, then T5 GEC cleanup."""
        # 1. Run local NLP rules (grammar_engine)
        try:
            from grammar_engine import process_sentence
            raw_tokens = text.split()
            rule_based_result = process_sentence(raw_tokens)
            if not rule_based_result:
                rule_based_result = text.strip().lower()
        except Exception as e:
            print(f"[GEC] Rule engine error: {e}")
            rule_based_result = text.strip().lower()

        # 2. Run T5 inference to polish the result
        if self._gec_ready and self._gec_model is not None:
            try:
                import torch
                prompt = f"gec: {rule_based_result}"
                inputs = self._gec_tokenizer(
                    prompt, return_tensors="pt",
                    max_length=128, truncation=True
                )
                with torch.no_grad():
                    outputs = self._gec_model.generate(
                        inputs["input_ids"],
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                result = self._gec_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Ensure first letter is capitalised
                return result[0].upper() + result[1:] if result else rule_based_result.capitalize()
            except Exception as e:
                print(f"[GEC] Inference error: {e}")
                
        # Fallback: Just return rule engine result
        return rule_based_result[0].upper() + rule_based_result[1:] if rule_based_result else ""

    def manual_speak(self):
        if not self.word.strip():
            return
        if self._enhanced:
            return  # already enhanced — do nothing until new content added
        # Disable button immediately; run inference in background so UI stays responsive
        if hasattr(self, 'enhance_btn'):
            self.enhance_btn.configure(state="disabled", fg_color=self.card_alt,
                                       text="⏳ Enhancing…")
        raw_text = self.word.strip()

        def _run():
            corrected = self._gec_correct(raw_text)
            def _apply():
                self.word = corrected + " "
                self.set_target_word(self.word)
                self._enhanced = True
                if hasattr(self, 'enhance_btn'):
                    self.enhance_btn.configure(
                        state="disabled",
                        fg_color=self.card_alt,
                        text="\U0001fa84 Enhance Syntax"
                    )
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

    def round_corners_pil(self, im, rad):
        from PIL import ImageDraw, Image
        circle = Image.new('L', (rad * 2, rad * 2), 0)
        draw = ImageDraw.Draw(circle)
        draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
        alpha = Image.new('L', im.size, 255)
        w, h = im.size
        alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
        alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
        alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
        alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
        im.putalpha(alpha)
        return im

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.video_label.configure(text="❌ Camera not found!", image="")
            self.root.after(1000, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = self._process_frame(mp_image)

        frame = self.apply_video_effects(frame, hand_result)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        from PIL import Image
        img = Image.fromarray(rgb)
        
        # The video frame is sized to maintain 4:3, so simple fill — no bezels, no distortion, fast.
        v_width = self.video_frame.winfo_width()
        v_height = self.video_frame.winfo_height()
        if v_width < 10 or v_height < 10:
            v_width, v_height = 640, 480

        img = img.resize((v_width, v_height))
        img = self.round_corners_pil(img, 20)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(v_width, v_height))
        self.video_label.configure(text="", image=imgtk)


        self.root.after(15, self.update_video)

    def _process_frame(self, mp_image):
        now = time.time()
        features, hand_detected, hand_result = self._extract_word_features(mp_image)

        if not hand_detected:
            self.stable_frames = 0
            self.current_stable_letter = ""
            self.word_recording = False
            self.word_raw_buffer.clear()
            if hasattr(self, 'word_progress'):
                self.word_progress.set(0)
            self.conf_display.configure(text="Confidence: 0%")
            self.draw_confidence_ring(0)
            if hasattr(self, 'auto_mode_status'):
                self.auto_mode_status.configure(text="Waiting for sign...")
            if len(self.letter_buffer) > 0 and (now - self.last_seen_time) > self.PAUSE_TIME:
                self.letter_buffer = []
            return hand_result

        # --- LETTER EVALUATION ---
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
                # EMIT LETTER
                if len(self.letter_buffer) == 0 or letter_label != self.letter_buffer[-1]:
                    self.letter_buffer.append(letter_label)
                    self.word += letter_label
                    self.set_target_word(self.word)
                    self._mark_content_changed()
                self.last_seen_time = now
                
                # RESET WORD RECORDING (User is holding a steady letter!)
                self.word_recording = False
                self.word_raw_buffer.clear()
                if hasattr(self, 'word_progress'):
                    self.word_progress.set(0)
                if hasattr(self, 'auto_mode_status'):
                    self.auto_mode_status.configure(text=f"Letter: {letter_label}")
                
                self.conf_display.configure(text=f"Confidence: {int(letter_conf * 100)}%")
                self.draw_confidence_ring(int(letter_conf * 100))
                return hand_result
        else:
            self.stable_frames = 0
            
        # --- WORD EVALUATION (Explicit Window) ---
        if now < self.word_cooldown_until:
            if hasattr(self, 'auto_mode_status'):
                self.auto_mode_status.configure(text="Cooldown...")
            self.conf_display.configure(text=f"Confidence: {int(letter_conf * 100)}%")
            self.draw_confidence_ring(int(letter_conf * 100))
            return hand_result

        if not self.word_recording:
            self.word_recording = True
            self.word_record_start = now
            self.word_raw_buffer = []
            if hasattr(self, 'word_progress'):
                self.word_progress.set(0)
            
        elapsed = now - self.word_record_start
        progress = min(1.0, elapsed / self.RECORD_DURATION)
        if hasattr(self, 'word_progress'):
            self.word_progress.set(progress)
        
        self.word_raw_buffer.append(features)

        if elapsed < self.RECORD_DURATION:
            if hasattr(self, 'auto_mode_status'):
                self.auto_mode_status.configure(text=f"🔴 Recording Word… {elapsed:.1f} / {self.RECORD_DURATION:.1f}s")
            self.conf_display.configure(text=f"Confidence: {int(letter_conf * 100)}%")
            self.draw_confidence_ring(int(letter_conf * 100))
        else:
            # End of window. Evaluate word.
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
                    # Emit Word!
                    self.conf_display.configure(text=f"Word Conf: {int(word_confidence * 100)}%")
                    self.draw_confidence_ring(int(word_confidence * 100))
                    if hasattr(self, 'auto_mode_status'):
                        self.auto_mode_status.configure(text=f"✅ {word_label}")
                    
                    self.word += word_label + " "
                    self.set_target_word(self.word)
                    self._mark_content_changed()

                    self.word_cooldown_until = now + 1.5
                    return hand_result

            # If we didn't emit a Word
            if hasattr(self, 'auto_mode_status'):
                self.auto_mode_status.configure(text="Not recognised.")
            
            # Start recording the next window immediately
            self.word_recording = True
            self.word_record_start = now
            self.word_raw_buffer = [features]
            if hasattr(self, 'word_progress'):
                self.word_progress.set(0)

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
    app_root = ctk.CTk()
    app = SignLanguageApp(app_root)
    app_root.protocol("WM_DELETE_WINDOW", app.on_close)
    app_root.mainloop()