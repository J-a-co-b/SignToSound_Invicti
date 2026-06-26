# pi_runner.py — Headless runner (Raspberry Pi 3B+ / NVIDIA Jetson Nano)
import os, sys, time, json
import cv2
import numpy as np
import joblib
import pyttsx3
import multiprocessing
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

try:
    import ST7789
    HAS_DISPLAY = True
except ImportError:
    print("WARNING: ST7789 not found. Running without display.")
    HAS_DISPLAY = False

# ── Jetson Nano: detect hardware ────────────────────────────────────────────────
def _is_jetson():
    try:
        with open("/proc/device-tree/model") as f:
            return "jetson" in f.read().lower()
    except Exception:
        return False

IS_JETSON = _is_jetson()

from grammar_engine import process_sentence

# ── Constants ──────────────────────────────────────
POSE_IDXS      = [11, 12, 13, 14, 15, 16, 23, 24]
WORD_FRAMES    = 30
DISP_W, DISP_H = 320, 240
# Jetson captures at higher resolution; Pi stays at 320x240 to keep CPU load low
CAP_W  = 640 if IS_JETSON else 320
CAP_H  = 480 if IS_JETSON else 240
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))

# ── Jetson CSI camera pipeline (IMX219 / Raspberry Camera v2) ──────────────────
def _gstreamer_pipeline(framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAP_W}, height=(int){CAP_H}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method=2 ! "
        f"video/x-raw, width=(int){CAP_W}, height=(int){CAP_H}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def _open_camera():
    """On Jetson: try CSI (GStreamer) first, fall back to USB. On Pi: USB only."""
    if IS_JETSON:
        cap = cv2.VideoCapture(_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("Camera: CSI via GStreamer")
            return cap
        cap.release()
        print("CSI not found, falling back to USB camera.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    print("Camera: USB")
    return cap

# ── TTS Worker ─────────────────────────────────────
def tts_worker(conn):
    while True:
        try:
            text = conn.recv()
            if text == "STOP":
                break
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except EOFError:
            break
        except Exception as e:
            print(f"TTS Error: {e}")

# ── Main Device ────────────────────────────────────
class PiSignToSound:
    def __init__(self):
        print("🔧 Initializing SignToSound Device...")

        # SPI TFT Display
        # Jetson Nano J41: DC=22 (GPIO25), RST=11 (GPIO17), BL=32 (GPIO12)
        # Raspberry Pi:    DC=24 (GPIO8),  RST=25 (GPIO25), BL=18 (GPIO24)
        if HAS_DISPLAY:
            dc  = 22 if IS_JETSON else 24
            rst = 11 if IS_JETSON else 25
            bl  = 32 if IS_JETSON else 18
            self.disp = ST7789.ST7789(
                port=0, cs=0, dc=dc, rst=rst, backlight=bl,
                spi_speed_hz=80_000_000
            )
            self.disp.begin()
            self._show_splash("SignToSound\nStarting...")
        else:
            self.disp = None

        # TTS process
        self.parent_conn, child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(target=tts_worker, args=(child_conn,), daemon=True)
        self.proc.start()

        # Scalers
        self.scaler      = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
        self.word_scaler = joblib.load(os.path.join(SCRIPT_DIR, "word_scaler.pkl"))

        # Label maps
        with open(os.path.join(SCRIPT_DIR, "word_label_map.json")) as f:
            self.word_labels = json.load(f)
        self.actions = np.array([
            'A','B','C','D','E','F','G','H','I',
            'K','L','M','N','O','P','Q','R','S',
            'T','U','V','W','X','Y'
        ])

        # TFLite — Letter model
        # Jetson Nano has 4 Cortex-A57 cores; Pi 3B+ has 4 Cortex-A53 cores
        _threads = 4
        self.lt_interp = tflite.Interpreter(
            model_path=os.path.join(SCRIPT_DIR, "sign_language_model.tflite"),
            num_threads=_threads)
        self.lt_interp.allocate_tensors()
        self.lt_in  = self.lt_interp.get_input_details()[0]['index']
        self.lt_out = self.lt_interp.get_output_details()[0]['index']

        # TFLite — Word model
        self.wd_interp = tflite.Interpreter(
            model_path=os.path.join(SCRIPT_DIR, "word_model.tflite"),
            num_threads=_threads)
        self.wd_interp.allocate_tensors()
        self.wd_in  = self.wd_interp.get_input_details()[0]['index']
        self.wd_out = self.wd_interp.get_output_details()[0]['index']

        # MediaPipe
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=os.path.join(SCRIPT_DIR, "hand_landmarker.task")),
                num_hands=2))
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=os.path.join(SCRIPT_DIR, "pose_landmarker.task")),
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4))

        # Camera — auto-selects CSI (Jetson) or USB
        self.cap = _open_camera()

        # State
        self.mode                = "WORD"
        self.sentence            = ""
        self.status_text         = "Show your hands..."
        self.last_prediction     = "—"
        self.last_confidence     = 0.0

        # Letter state
        self.pred_buffer         = deque(maxlen=5)
        self.stable_letter       = ""
        self.stable_frames       = 0
        self.letter_buffer       = []
        self.last_seen_time      = time.time()

        # Word state
        self.RECORD_DURATION     = 2.5
        self.word_raw_buffer     = []
        self.word_recording      = False
        self.word_record_start   = 0.0
        self.word_cooldown_until = 0.0

        print("✅ Ready!")
        if self.disp:
            self._show_splash("SignToSound\nReady!")
        time.sleep(1)

    def speak(self, text):
        if text.strip():
            try:
                self.parent_conn.send(text)
            except Exception:
                pass

    def _show_splash(self, text):
        if not self.disp:
            return
        img  = Image.new("RGB", (DISP_W, DISP_H), "#1A1B2F")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for i, line in enumerate(text.split("\n")):
            draw.text((DISP_W // 2 - 60, DISP_H // 2 - 10 + i * 20),
                      line, fill="#00FFCC", font=font)
        self.disp.display(img)

    def _extract_word_features(self, mp_image):
        hand_result  = self.hand_detector.detect(mp_image)
        pose_result  = self.pose_detector.detect(mp_image)
        hand_detected = bool(hand_result.hand_landmarks)

        slots = {0: np.zeros(63, np.float32), 1: np.zeros(63, np.float32)}
        if hand_result.hand_landmarks:
            for hand, cat in zip(hand_result.hand_landmarks, hand_result.handedness):
                side = 0 if cat[0].category_name == "Left" else 1
                pts  = np.array([[lm.x, lm.y, lm.z] for lm in hand], np.float32)
                pts -= pts[0]
                slots[side] = pts.flatten()

        pose_vec = np.zeros(24, np.float32)
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            pts = np.array([[lms[i].x, lms[i].y, lms[i].z]
                            for i in POSE_IDXS], np.float32)
            pose_vec = (pts - (pts[0] + pts[1]) / 2.0).flatten()

        return np.concatenate([slots[0], slots[1], pose_vec]), hand_detected

    def process_letter_mode(self, mp_image):
        result = self.hand_detector.detect(mp_image)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wx, wy, wz = hand[0].x, hand[0].y, hand[0].z
            lms = np.array(
                [c for lm in hand for c in [lm.x-wx, lm.y-wy, lm.z-wz]],
                np.float32).reshape(1, 63)

            if self.scaler:
                lms = self.scaler.transform(lms).astype(np.float32)

            self.lt_interp.set_tensor(self.lt_in, lms)
            self.lt_interp.invoke()
            pred = self.lt_interp.get_tensor(self.lt_out)

            self.pred_buffer.append(pred)
            avg    = np.mean(self.pred_buffer, axis=0)[0]
            cid    = np.argmax(avg)
            conf   = float(np.max(avg))
            letter = self.actions[cid]

            self.last_confidence = conf
            self.status_text = f"Conf: {int(conf*100)}%  Stable: {self.stable_frames}/5"

            if conf > 0.70:
                if letter == self.stable_letter:
                    self.stable_frames += 1
                else:
                    self.stable_letter = letter
                    self.stable_frames = 1
                if self.stable_frames == 5:
                    self.last_prediction = letter
                    if not self.letter_buffer or letter != self.letter_buffer[-1]:
                        self.letter_buffer.append(letter)
                        self.sentence += letter
                    self.last_seen_time = time.time()
            else:
                self.stable_frames = 0
        else:
            self.stable_frames = 0
            self.stable_letter = ""
            self.status_text   = "No hand detected"
            if self.letter_buffer and (time.time() - self.last_seen_time) > 1.5:
                self.speak(self.sentence.strip())
                self.letter_buffer = []

    def process_word_mode(self, mp_image):
        now = time.time()
        if now < self.word_cooldown_until:
            return

        if not self.word_recording:
            feat, hand_detected = self._extract_word_features(mp_image)
            if hand_detected:
                self.word_recording    = True
                self.word_record_start = now
                self.word_raw_buffer   = [feat]
                self.status_text       = "🔴 Recording..."
                self.last_prediction   = "…"
            else:
                self.status_text = "Show your hands..."
            return

        elapsed = now - self.word_record_start
        self.status_text = f"🔴 {elapsed:.1f} / {self.RECORD_DURATION:.1f}s"
        feat, _ = self._extract_word_features(mp_image)
        self.word_raw_buffer.append(feat)

        if elapsed >= self.RECORD_DURATION:
            self.word_recording = False
            raw = self.word_raw_buffer
            self.word_raw_buffer = []

            if len(raw) < 5:
                self.status_text = "Too few frames — try again"
                self.word_cooldown_until = now + 1.0
                return

            idxs = np.linspace(0, len(raw) - 1, WORD_FRAMES, dtype=int)
            seq  = np.array([raw[i] for i in idxs], np.float32)

            if self.word_scaler:
                seq = self.word_scaler.transform(
                    seq.reshape(1, -1)).reshape(1, WORD_FRAMES, 150).astype(np.float32)
            else:
                seq = seq.reshape(1, WORD_FRAMES, 150)

            self.wd_interp.set_tensor(self.wd_in, seq)
            self.wd_interp.invoke()
            probs = self.wd_interp.get_tensor(self.wd_out)[0]

            cid   = int(np.argmax(probs))
            conf  = float(probs[cid])
            label = self.word_labels[str(cid)]

            entropy     = -np.sum(probs * np.log(probs + 1e-9))
            max_entropy = np.log(len(probs))
            if entropy > 0.75 * max_entropy or conf < 0.80:
                self.status_text = "Uncertain — sign more clearly"
                self.word_cooldown_until = now + 1.5
                return

            self.last_prediction = label
            self.last_confidence = conf
            self.sentence       += label + " "
            self.sentence        = process_sentence(self.sentence.split()) + " "
            self.status_text     = f"✅ {label}  ({int(conf*100)}%)"
            self.speak(label)
            self.word_cooldown_until = now + 1.5

    def render_dashboard(self, rgb_frame):
        img  = Image.new("RGB", (DISP_W, DISP_H), "#1A1B2F")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Live camera preview (top-left)
        preview = Image.fromarray(cv2.resize(rgb_frame, (140, 105)))
        img.paste(preview, (5, 5))

        # Info panel (top-right)
        draw.rectangle([(150, 5), (315, 115)], fill="#252641", outline="#4B5085")
        draw.text((158, 10), f"Mode: {self.mode}",         fill="#00FFCC", font=font)
        draw.text((158, 28), f"Sign: {self.last_prediction}", fill="#FFD700", font=font)
        draw.text((158, 46), self.status_text[:22],         fill="#BDC3C7", font=font)

        # Recording progress bar
        if self.mode == "WORD" and self.word_recording:
            elapsed = time.time() - self.word_record_start
            prog_w  = int((min(elapsed, self.RECORD_DURATION) / self.RECORD_DURATION) * 140)
            draw.rectangle([(158, 68), (298, 80)],              fill="#333355")
            draw.rectangle([(158, 68), (158 + prog_w, 80)],     fill="#FF4D4D")

        # Sentence bar (bottom)
        draw.rectangle([(5, 120), (315, 235)], fill="#1E2038", outline="#4B5085")
        draw.text((12, 124), "Sentence:", fill="#FFD700", font=font)

        words = self.sentence.split()
        lines, cur = [], []
        for w in words:
            if len(" ".join(cur + [w])) > 40:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        for i, line in enumerate(lines[-5:]):
            draw.text((12, 140 + i * 18), line, fill="#FFFFFF", font=font)

        return img

    def run(self):
        print("▶️  Running. Press Ctrl+C to stop.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame    = cv2.flip(frame, 1)
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                if self.mode == "LETTER":
                    self.process_letter_mode(mp_image)
                else:
                    self.process_word_mode(mp_image)

                if self.disp:
                    self.disp.display(self.render_dashboard(rgb))

                time.sleep(0.01)

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print("\nShutting down...")
        try:
            self.parent_conn.send("STOP")
        except Exception:
            pass
        self.cap.release()
        if self.disp:
            self.disp.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    PiSignToSound().run()
