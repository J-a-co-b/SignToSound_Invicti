import os
import platform as _platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLIB_SKIP_ARGB_VISUALS'] = '1'

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
from tensorflow.keras.models import Sequential, Model, load_model
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

# Patch customtkinter's FontManager on Jetson ARM64, AFTER importing it.
# CTk bundles a Roboto TTF and registers its full glyph set with the X server;
# Jetson's X11 RENDER extension rejects that registration with
# "BadLength (RenderAddGlyphs)", crashing the process the moment any CTk
# widget is drawn. Making load_font a no-op stops CTk from ever registering
# Roboto, so widgets fall back to Tk's default system font instead.
# Scanning sys.modules (rather than guessing a submodule path) finds
# FontManager regardless of which internal module customtkinter put it in.
if _platform.machine() == 'aarch64':
    import sys
    _patched_any = False
    for _mod_name, _mod in list(sys.modules.items()):
        if _mod_name and _mod_name.startswith('customtkinter') and hasattr(_mod, 'FontManager'):
            _mod.FontManager.load_font = classmethod(lambda cls, *a, **kw: False)
            print(f"[Jetson] Patched FontManager in {_mod_name}")
            _patched_any = True
    if not _patched_any:
        print("[Jetson] WARNING: could not find customtkinter.FontManager to patch")

from PIL import Image, ImageTk
import h5py

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# Pose landmark indices for word-mode feature extraction
# ==================================================
POSE_IDXS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
WORD_FRAMES = 30  # frames to buffer before word prediction


class _NumpyScaler:
    """Minimal StandardScaler replacement using only numpy.

    Avoids importing sklearn at runtime: on Jetson/aarch64, unpickling a
    sklearn StandardScaler via joblib.load() pulls in sklearn's native
    extensions, which crash with "cannot allocate memory in static TLS
    block" once TensorFlow has already claimed the limited static TLS
    space. Loading mean_/scale_ straight out of a plain .npz sidesteps
    sklearn entirely.
    """
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def _load_scaler(npz_path, pkl_path):
    """Load a fitted StandardScaler-like object, preferring the sklearn-free
    .npz format. Falls back to joblib/.pkl only if no .npz is present, and
    never lets a sklearn import failure crash the app."""
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return _NumpyScaler(data["mean"], data["scale"])
    if os.path.exists(pkl_path):
        try:
            import joblib
            return joblib.load(pkl_path)
        except Exception as exc:
            print(f"[WARN] Failed to load scaler '{pkl_path}' via joblib/sklearn: {exc}")
    return None


# ==================================================
# Robust weight loader — handles TF2 Keras and Keras 3 formats
# ==================================================
def _load_weights_smart(model, filepath):
    """Try every known .h5 weight format so the app works regardless of
    which Keras version saved the file.

    Strategy 1 – TF2 Keras load_weights  (old .h5 weights-only format)
    Strategy 2 – h5py Keras-3 format     (/layers/{name}/vars/{0,1,...})
    Strategy 3 – h5py flat format        (/{layer_name}/{weight_name})
    """
    # Store errors as strings — Python 3 deletes 'except ... as e' variables
    # after the except block exits, causing UnboundLocalError if referenced later.
    e1 = e2 = "not reached"

    # ── 1. TF2 Keras load_weights ──────────────────────────────────────
    try:
        model.load_weights(filepath)
        print(f"[INFO] load_weights() OK  ← {filepath}")
        return
    except Exception as _exc:
        e1 = str(_exc)
        print(f"[WARN] load_weights failed: {e1}")

    # ── 2 & 3. h5py direct read ────────────────────────────────────────
    try:
        with h5py.File(filepath, 'r') as hf:

            # ── 2. Keras 3 format: /layers/{name}/vars/{0,1,...} ───────
            if 'layers' in hf:
                loaded, skipped = 0, 0
                layers_grp = hf['layers']
                for layer in model.layers:
                    if not layer.weights:
                        continue
                    lname = layer.name
                    if lname in layers_grp and 'vars' in layers_grp[lname]:
                        vg = layers_grp[lname]['vars']
                        w = [np.array(vg[str(i)]) for i in range(len(vg))]
                        if w:
                            try:
                                layer.set_weights(w)
                                loaded += 1
                            except ValueError as _se:
                                print(f"[WARN] Shape mismatch layer '{lname}': {_se}")
                                skipped += 1
                if loaded and skipped == 0:
                    print(f"[INFO] h5py Keras-3 format OK — {loaded} layers  ← {filepath}")
                    return
                if loaded and skipped:
                    raise ValueError(
                        f"Keras-3 format: {loaded} layers loaded but {skipped} had shape mismatches. "
                        f"The word model architecture in gui_app.py doesn't match the trained model. "
                        f"Re-run with _infer_word_arch() to auto-detect."
                    )
                print("[WARN] /layers group found but no layer names matched; trying flat format…")

            # ── 3. Flat format: /{layer_name}/{weight_name} ──
            loaded = 0
            for layer in model.layers:
                if not layer.weights:
                    continue
                if layer.name in hf:
                    wg = hf[layer.name]
                    w = [np.array(wg[k]) for k in sorted(wg.keys())]
                    if w:
                        try:
                            layer.set_weights(w)
                            loaded += 1
                        except ValueError as _se:
                            print(f"[WARN] Shape mismatch layer '{layer.name}': {_se}")
            if loaded:
                print(f"[INFO] h5py flat format OK — {loaded} layers  ← {filepath}")
                return

            # Dump top-level structure to help diagnose unknown formats
            top_keys = list(hf.keys())
            all_paths: list = []
            hf.visititems(lambda n, _: all_paths.append(n))
            raise ValueError(
                f"Unrecognised HDF5 structure.\n"
                f"  Top-level keys : {top_keys}\n"
                f"  All paths (≤30): {all_paths[:30]}"
            )
    except Exception as _exc:
        e2 = str(_exc)
        raise RuntimeError(
            f"All weight-loading strategies failed for '{filepath}':\n"
            f"  1. TF2 load_weights : {e1}\n"
            f"  2+3. h5py           : {e2}"
        )


def _h5_all_arrays(group):
    """Recursively collect every dataset in an h5py group, sorted by path.
    Handles both flat (vars/0) and Keras-3 nested (layers/sub/vars/0) structures."""
    result = []
    group.visititems(
        lambda name, obj: result.append((name, np.array(obj)))
        if isinstance(obj, h5py.Dataset) else None
    )
    return [v for _, v in sorted(result)]   # alphabetical path sort


def _infer_word_arch(filepath):
    """Read weight shapes directly from the sep_conv / dense layers in the file —
    NOT from the BatchNorm layers, which have naming-counter issues.
    Returns (f1, f2, f3, d1).  Falls back to (64, 64, 32, 64) on any error."""
    defaults = (64, 64, 32, 64)
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'layers' not in hf:
                return defaults
            lg = hf['layers']

            def _conv_out_filters(conv_name):
                """Read output-filter count from a SeparableConv1D saved in Keras-3 format.
                Alphabetical sort → [depthwise_kernel, pointwise_kernel, (bias)].
                pointwise kernel shape: (1, in_ch*dm, out_filters) or (in_ch*dm, out_filters)."""
                if conv_name not in lg:
                    return None
                arrays = _h5_all_arrays(lg[conv_name])
                # Need at least 2 arrays (depthwise + pointwise kernels)
                if len(arrays) < 2:
                    return None
                pw_kernel = arrays[1]       # index 1 = pointwise kernel
                return int(pw_kernel.shape[-1])

            def _dense_out(dense_name):
                """kernel shape[-1] → number of Dense output units."""
                if dense_name not in lg:
                    return None
                arrays = _h5_all_arrays(lg[dense_name])
                if not arrays:
                    return None
                return int(arrays[0].shape[-1])   # kernel is first array

            f1 = _conv_out_filters('separable_conv1d') or defaults[0]
            f2 = _conv_out_filters('separable_conv1d_1') or defaults[1]
            f3 = _conv_out_filters('separable_conv1d_2') or defaults[2]
            d1 = _dense_out('dense')             or defaults[3]
            print(f"[INFO] Inferred word model arch: conv({f1})-conv({f2})-conv({f3})-dense({d1})")
            return f1, f2, f3, d1
    except Exception as _exc:
        print(f"[WARN] _infer_word_arch failed ({_exc}), using defaults {defaults}")
        return defaults


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
        # 1st attempt: load_model handles files saved with model.save()
        # 2nd attempt: build Sequential + _load_weights_smart (TF2 or Keras-3 weights format)
        try:
            self.model = load_model('sign_language_model.weights.h5', compile=False)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("[INFO] Letter model loaded via load_model()")
        except Exception:
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
            _load_weights_smart(self.model, 'sign_language_model.weights.h5')
        self.actions = np.array(['A','B','C','D','E','F','G','H','I',
                                  'K','L','M','N','O','P','Q','R','S',
                                  'T','U','V','W','X','Y'])

        # --- LOAD WORD MODEL ---
        with open("word_label_map.json") as f:
            self.word_labels = json.load(f)
        self.word_list = [self.word_labels[str(i)] for i in range(len(self.word_labels))]
        n_classes = len(self.word_labels)

        def _build_word_model(f1=64, f2=64, f3=32, d1=64):
            """Build word model with filter/unit counts read from the saved file."""
            _inp = Input(shape=(WORD_FRAMES, 150), name="landmarks")
            _x = SeparableConv1D(f1, kernel_size=3, padding="same",
                                depthwise_regularizer=l2(1e-4),
                                pointwise_regularizer=l2(1e-4),
                                name="sep_conv1")(_inp)
            _x = BatchNormalization()(_x)
            _x = Activation("relu")(_x)
            _x = Dropout(0.25)(_x)
            _x = SeparableConv1D(f2, kernel_size=5, padding="same",
                                depthwise_regularizer=l2(1e-4),
                                pointwise_regularizer=l2(1e-4),
                                name="sep_conv2")(_x)
            _x = BatchNormalization()(_x)
            _x = Activation("relu")(_x)
            _x = Dropout(0.25)(_x)
            _x = SeparableConv1D(f3, kernel_size=7, padding="same",
                                depthwise_regularizer=l2(1e-4),
                                pointwise_regularizer=l2(1e-4),
                                name="sep_conv3")(_x)
            _x = BatchNormalization()(_x)
            _x = Activation("relu")(_x)
            _x = Dropout(0.20)(_x)
            _x = GlobalAveragePooling1D()(_x)
            _x = Dense(d1, activation="relu", kernel_regularizer=l2(1e-4))(_x)
            _x = Dropout(0.25)(_x)
            _out = Dense(n_classes, activation="softmax", name="predictions")(_x)
            return Model(_inp, _out, name="SignToSound_Word")

        # ── Word model loading ─────────────────────────────────────────────────
        # Try load_model first (full-model save format).
        # Otherwise: positional loading from the Keras-3 .h5 file.
        #
        # WHY POSITIONAL?
        # The letter model is built first, consuming the global Keras layer-name
        # counters (e.g. batch_normalization=0, batch_normalization_1=1).
        # The word model's BN layers then get names like batch_normalization_2/3/4
        # while the file (saved in a fresh Keras-3 session) has them as
        # batch_normalization/batch_normalization_1/batch_normalization_2.
        # Name matching therefore assigns the wrong weights → shape mismatch.
        # Positional matching zips file layers to model layers by topology order,
        # regardless of their names.
        #
        # File layer topology order (must match _build_word_model architecture).
        # Names are Keras's auto-generated names from train_words.py (which did not
        # pass explicit `name=` to the SeparableConv1D layers), NOT the
        # "sep_conv1/2/3" names _build_word_model uses locally — those only need to
        # match positionally, not by name.
        _WORD_FILE_TOPOLOGY = [
            'separable_conv1d',   'batch_normalization',
            'separable_conv1d_1', 'batch_normalization_1',
            'separable_conv1d_2', 'batch_normalization_2',
            'dense',              'dense_1',
        ]

        def _load_word_positional(wmodel, wpath):
            """Load word model weights from Keras-3 h5 by topology position."""
            with h5py.File(wpath, 'r') as hf:
                if 'layers' not in hf:
                    raise ValueError("No /layers group in word_model.weights.h5")
                lg = hf['layers']
                wt_layers = [l for l in wmodel.layers if l.weights]
                if len(wt_layers) != len(_WORD_FILE_TOPOLOGY):
                    raise ValueError(
                        f"Expected {len(_WORD_FILE_TOPOLOGY)} weight layers, "
                        f"got {len(wt_layers)}: {[l.name for l in wt_layers]}"
                    )
                for ml, fname in zip(wt_layers, _WORD_FILE_TOPOLOGY):
                    if fname not in lg:
                        raise ValueError(f"Layer '{fname}' missing from file")
                    ml.set_weights(_h5_all_arrays(lg[fname]))
            print(f"[INFO] Word model loaded positionally ({len(_WORD_FILE_TOPOLOGY)} layers)")

        try:
            self.word_model = load_model("word_model.weights.h5", compile=False)
            self.word_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            print("[INFO] Word model loaded via load_model()")
        except Exception:
            f1, f2, f3, d1 = _infer_word_arch("word_model.weights.h5")
            self.word_model = _build_word_model(f1, f2, f3, d1)
            self.word_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            # First try standard load_weights (handles old TF2 format)
            try:
                self.word_model.load_weights("word_model.weights.h5")
                print("[INFO] Word model loaded via load_weights()")
            except Exception:
                # Fall back to positional loading (handles Keras-3 format)
                _load_word_positional(self.word_model, "word_model.weights.h5")

        # --- LOAD SCALERS ---
        self.scaler = _load_scaler(
            os.path.join(os.getcwd(), "scaler.npz"),
            os.path.join(os.getcwd(), "scaler.pkl"),
        )
        self.word_scaler = _load_scaler(
            os.path.join(os.getcwd(), "word_scaler.npz"),
            os.path.join(os.getcwd(), "word_scaler.pkl"),
        )

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

        # --- MODE ---
        self.mode_var = ctk.StringVar(value="LETTER")

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

    def _update_video_column_width(self):
        win_h = self.root.winfo_height()
        win_w = self.root.winfo_width()
        if win_h < 50 or win_w < 50:
            return
        # Vertical overhead: title bar (~30) + top pady (20) + gap (10) + bottom bar (180) + bottom paddings (30)
        video_panel_h = max(300, win_h - 270)
        # Width to maintain 4:3, plus frame's own padding
        video_col_w = int(video_panel_h * 4 / 3) + 30
        # Never take more than 65% of window width
        video_col_w = min(video_col_w, int(win_w * 0.65))
        self.root.grid_columnconfigure(0, weight=0, minsize=video_col_w)

    def _on_root_configure(self, event):
        if event.widget is self.root:
            self._update_video_column_width()

    def build_ui(self):
        self.f_title = ("Segoe UI", 24, "bold")
        self.f_header = ("Segoe UI", 16, "bold")
        self.f_body = ("Segoe UI", 14)
        self.f_large = ("Segoe UI", 48, "bold")
        
        # Size the video column to maintain 4:3 aspect ratio based on available window height.
        # This keeps the camera view natural (no zoom) and lets the right panel expand freely.
        # Recomputed on every root <Configure>, not just once here: some window managers
        # (e.g. the Jetson's) apply root.state("zoomed") asynchronously, so winfo_height()
        # at this point can still report the pre-zoom geometry. Pinning column 0's minsize
        # to that stale value while row 0 (weight=1) later stretches to the real, taller
        # window leaves the video panel tall and narrow with huge letterbox bars.
        self.root.update_idletasks()
        self._update_video_column_width()
        self.root.grid_columnconfigure(1, weight=1)   # right panel expands
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0, minsize=180)
        self.root.bind("<Configure>", self._on_root_configure)
        
        # ── Video panel ──
        self.video_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color="#000000")
        self.video_frame.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Webcam Loading...", font=self.f_body, fg_color="transparent")
        self.video_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        # ── Info panel — all children packed, nothing ever clips ──
        self.info_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=self.card_color)
        self.info_frame.grid(row=0, column=1, padx=(10, 20), pady=(20, 10), sticky="nsew")

        # Mode toggle
        self.mode_seg = ctk.CTkSegmentedButton(
            self.info_frame, values=["LETTER", "WORD"],
            variable=self.mode_var,
            command=self._on_mode_change,
            font=self.f_header,
            selected_color=self.accent_primary,
            selected_hover_color="#4F46E5",
            unselected_color=self.card_alt,
            height=40
        )
        self.mode_seg.pack(pady=(20, 0), padx=20, fill="x")

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
            self.ring_canvas, text="0%", font=("Segoe UI", 26, "bold"),
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

        # Status area — either stability (letter mode) or recording status (word mode)
        self.status_area = ctk.CTkFrame(self.info_frame, fg_color="transparent", height=30)
        self.status_area.pack(pady=(4, 0), padx=20, fill="x")

        self.stable_display = ctk.CTkLabel(
            self.status_area, text="Stability: 0/5", font=self.f_body
        )
        self.stable_display.pack()  # shown in LETTER mode

        self.word_mode_status = ctk.CTkLabel(
            self.status_area, text="", font=self.f_body, text_color=self.text_secondary
        )
        # not packed initially — shown only in WORD mode

        self.word_progress = ctk.CTkProgressBar(
            self.info_frame, height=8, progress_color=self.accent_primary
        )
        self.word_progress.set(0)
        # not packed initially — shown only in WORD mode

        # ── Button row 1a: Space ──
        btn_space = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_space.pack(pady=(18, 0), padx=20, fill="x")
        ctk.CTkButton(btn_space, text="Space", font=self.f_body,
                      command=self.add_space, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x")

        # ── Button row 1b: Backspace / Delete Word / Clear ──
        btn_row1 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row1.pack(pady=(6, 0), padx=20, fill="x")
        ctk.CTkButton(btn_row1, text="Backspace", font=self.f_body,
                      command=self.delete_last, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(btn_row1, text="Del Word", font=self.f_body,
                      command=self.delete_word, fg_color=self.card_alt, width=0
                      ).pack(side="left", expand=True, fill="x", padx=4)
        ctk.CTkButton(btn_row1, text="Clear", font=self.f_body,
                      command=self.clear_word, fg_color="#E5484D", width=0
                      ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # ── Button row 2: Enhance (starts disabled until T5 loads) ──
        btn_row2 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row2.pack(pady=(8, 0), padx=20, fill="x")
        self.enhance_btn = ctk.CTkButton(
            btn_row2, text="Loading AI...", font=self.f_body,
            command=self.manual_speak, fg_color=self.card_alt,
            state="disabled", width=0
        )
        self.enhance_btn.pack(side="left", expand=True, fill="x")

        # ── Button row 3: Speak Out Loud ──
        btn_row3 = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        btn_row3.pack(pady=(8, 0), padx=20, fill="x")
        ctk.CTkButton(
            btn_row3, text="Speak Out Loud", font=self.f_body,
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
            font=("Segoe UI", 32, "bold"), text_color="#E3B341", justify="left", wraplength=900
        )
        self.word_display.place(x=20, y=60)
        
        self.word_info_label = ctk.CTkLabel(
            self.bottom_frame,
            text="LETTER MODE - Sign individual letters to spell words",
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

    def _on_mode_change(self, new_mode):
        self.word_raw_buffer.clear()
        self.prediction_buffer.clear()
        self.stable_frames = 0
        self.current_stable_letter = ""
        self.draw_confidence_ring(0)

        if new_mode == "WORD":
            self.word_recording    = False
            self.word_raw_buffer   = []
            self.word_cooldown_until = 0
            self.conf_display.configure(text="Confidence: -")
            self.stable_display.pack_forget()
            self.word_mode_status.pack()
            self.word_mode_status.configure(text="Show your hands to start recording")
            self.word_progress.pack(pady=(4, 0), padx=20, fill="x")
            self.word_progress.set(0)
            self.word_info_label.configure(text=f"WORD MODE - Show hands, hold sign for {self.RECORD_DURATION:.0f}s, auto-predicts")
        else:
            self.conf_display.configure(text="Confidence: 0%")
            self.stable_display.configure(text="Stability: 0/5")
            self.word_progress.pack_forget()
            self.word_mode_status.pack_forget()
            self.stable_display.pack()
            self.word_info_label.configure(text="LETTER MODE - Sign individual letters to spell words")

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
                text="Enhance Syntax"
            )

    def _gec_mark_unavailable(self):
        if hasattr(self, 'enhance_btn'):
            self.enhance_btn.configure(
                text="Enhance (basic)",
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
                                       text="Enhancing...")
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
                        text="Enhance Syntax"
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
        # The alpha mask only depends on (size, rad), which is static frame to
        # frame (panel size only changes on window resize) — cache it instead
        # of rebuilding it with 4 paste() calls on every single video frame.
        from PIL import ImageDraw, Image
        cache_key = (im.size, rad)
        if getattr(self, '_corner_mask_key', None) != cache_key:
            circle = Image.new('L', (rad * 2, rad * 2), 0)
            draw = ImageDraw.Draw(circle)
            draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
            w, h = im.size
            alpha = Image.new('L', im.size, 255)
            alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
            alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
            alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
            alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
            self._corner_mask_key = cache_key
            self._corner_mask = alpha
        im.putalpha(self._corner_mask)
        return im

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.video_label.configure(text="Camera not found!", image="")
            self.root.after(1000, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        current_mode = self.mode_var.get()
        hand_result = None

        if current_mode == "LETTER":
            hand_result = self._process_letter_mode(mp_image)
        else:
            hand_result = self._process_word_mode(mp_image)

        frame = self.apply_video_effects(frame, hand_result)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        from PIL import Image
        img = Image.fromarray(rgb)

        v_width = self.video_frame.winfo_width()
        v_height = self.video_frame.winfo_height()
        if v_width < 10 or v_height < 10:
            v_width, v_height = 640, 480

        # Scale to fit inside the panel preserving aspect ratio (no stretch), then
        # letterbox onto a black canvas of the panel's exact size. The panel isn't
        # guaranteed to be exactly 4:3 (window-manager geometry/zoom timing differs
        # across platforms), so a hard resize to (v_width, v_height) can distort the
        # image — this keeps the feed undistorted regardless of panel shape.
        fw, fh = img.size
        scale = min(v_width / fw, v_height / fh)
        new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        img = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (v_width, v_height), (0, 0, 0))
        canvas.paste(img, ((v_width - new_w) // 2, (v_height - new_h) // 2))
        img = canvas
        img = self.round_corners_pil(img, 20)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(v_width, v_height))
        self.video_label.configure(text="", image=imgtk)


        self.root.after(15, self.update_video)

    def _process_letter_mode(self, mp_image):
        result = self.hand_detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
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

            # Direct call instead of .predict(): predict() rebuilds a tf.data
            # pipeline on every invocation, which dominates runtime for
            # single-frame inference in a tight loop — calling the model
            # directly skips that overhead entirely.
            prediction = self.model(landmarks, training=False).numpy()
            self.prediction_buffer.append(prediction)

            avg_pred = np.mean(self.prediction_buffer, axis=0)[0]
            class_id = np.argmax(avg_pred)
            confidence = np.max(avg_pred)
            letter = self.actions[class_id]

            self.conf_display.configure(text=f"Confidence: {int(confidence * 100)}%")
            self.draw_confidence_ring(int(confidence * 100))

            if confidence > self.CONFIDENCE_THRESHOLD:
                if letter == self.current_stable_letter:
                    self.stable_frames += 1
                else:
                    self.current_stable_letter = letter
                    self.stable_frames = 1
                self.stable_display.configure(text=f"Stability: {self.stable_frames}/5")

                if self.stable_frames == self.REQUIRED_FRAMES:
                    if len(self.letter_buffer) == 0 or letter != self.letter_buffer[-1]:
                        self.letter_buffer.append(letter)
                        self.word += letter
                        self.set_target_word(self.word)
                        self._mark_content_changed()
                    self.last_seen_time = time.time()
            else:
                self.stable_frames = 0
        else:
            self.stable_frames = 0
            self.current_stable_letter = ""
            self.conf_display.configure(text="Confidence: 0%")
            self.stable_display.configure(text="Stability: 0/5")
            self.draw_confidence_ring(0)
            if len(self.letter_buffer) > 0 and (time.time() - self.last_seen_time) > self.PAUSE_TIME:
                self.letter_buffer = []
        return result

    def _process_word_mode(self, mp_image):
        now = time.time()
        features, hand_detected, hand_result = self._extract_word_features(mp_image)

        if now < self.word_cooldown_until:
            return hand_result

        if not self.word_recording:
            if hand_detected:
                self.word_recording    = True
                self.word_record_start = now
                self.word_raw_buffer   = [features]
                self.word_progress.set(0)
                self.word_mode_status.configure(text=f"Recording... 0.0 / {self.RECORD_DURATION:.1f}s")
            else:
                self.word_progress.set(0)
                self.word_mode_status.configure(text="Show your hands to start recording")
                self.conf_display.configure(text="Confidence: -")
                self.draw_confidence_ring(0)
            return hand_result

        elapsed = now - self.word_record_start
        progress = min(1.0, elapsed / self.RECORD_DURATION)
        self.word_progress.set(progress)
        self.word_mode_status.configure(text=f"Recording... {elapsed:.1f} / {self.RECORD_DURATION:.1f}s")

        self.word_raw_buffer.append(features)

        if elapsed >= self.RECORD_DURATION:
            self.word_recording = False
            raw = self.word_raw_buffer
            self.word_raw_buffer = []

            n_collected = len(raw)
            if n_collected < 5:
                self.word_mode_status.configure(text="Too few frames — try again")
                self.word_cooldown_until = now + 1.0
                return hand_result

            idxs = np.linspace(0, n_collected - 1, WORD_FRAMES, dtype=int)
            seq  = np.array([raw[i] for i in idxs], dtype=np.float32)

            if self.word_scaler is not None:
                flat = self.word_scaler.transform(seq.reshape(1, -1))
                seq  = flat.reshape(1, WORD_FRAMES, 150)
            else:
                seq = seq.reshape(1, WORD_FRAMES, 150)

            probs     = self.word_model(seq, training=False).numpy()[0]
            class_id  = int(np.argmax(probs))
            confidence = float(probs[class_id])
            word_label = self.word_labels[str(class_id)]

            entropy     = -np.sum(probs * np.log(probs + 1e-9))
            max_entropy = np.log(len(probs))
            if entropy > 0.75 * max_entropy:
                self.conf_display.configure(text="Uncertain — try again")
                self.word_mode_status.configure(text="Not recognised — sign more clearly")
                self.word_cooldown_until = now + 1.5
                self.draw_confidence_ring(0)
                return hand_result

            self.last_word_prediction  = word_label
            self.last_word_confidence  = confidence

            self.conf_display.configure(text=f"Confidence: {int(confidence * 100)}%")
            self.draw_confidence_ring(int(confidence * 100))
            self.word_mode_status.configure(
                text=f"{word_label}" if confidence > 0.80 else f"{word_label} - low confidence"
            )
            self.word_progress.set(1.0)

            if confidence > 0.80:
                self.word += word_label + " "
                self.set_target_word(self.word)
                self._mark_content_changed()

            self.word_cooldown_until = now + 1.5
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