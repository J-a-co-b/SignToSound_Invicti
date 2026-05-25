import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import mediapipe as mp

# ==================================================
# 1. LOAD MODEL & CLASSES
# ==================================================
model = load_model("sign_language_model.keras")

DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")
actions = np.array(sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and len(d) == 1 and d.isalpha()
]))

print("✅ Model loaded")
print("Classes:", actions)

# ==================================================
# 2. MEDIAPIPE HAND LANDMARKER
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ==================================================
# 3. PREDICTION SMOOTHING
# ==================================================
prediction_buffer = deque(maxlen=15)

# ==================================================
# 4. WEBCAM (WINDOWS FIX)
# 3. PREDICTION SMOOTHING & STABILITY LOGIC
# ==================================================
prediction_buffer = deque(maxlen=5) # Reduced from 15 to 5 to avoid lag

current_stable_letter = ""
stable_frames = 0
CONFIDENCE_THRESHOLD = 0.85
REQUIRED_FRAMES = 10 # Frames needed to lock in a letter

# ==================================================
# 4. LETTER BUFFERING VARIABLES
# ==================================================
letter_buffer = []
last_seen_time = time.time()
word = ""
PAUSE_TIME = 2  # seconds to detect word end

# ==================================================
# 5. WEBCAM
# ==================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible. Check camera permissions or index.")

print("🎥 Webcam started. Press 'q' to quit.")

# ==================================================
# 5. LOOP
# 6. LOOP
# ==================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32
        ).flatten()

        # SAME normalization as training
    # =============================================
    # HAND DETECTED
    # =============================================
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # --- NEW: Make coordinates relative to the wrist ---
        wrist_x = hand[0].x
        wrist_y = hand[0].y
        wrist_z = hand[0].z

        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

        landmarks = np.array(landmarks, dtype=np.float32)

        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        landmarks = landmarks.reshape(1, 63)

        # --- PREDICT ---
        prediction = model.predict(landmarks, verbose=0)
        prediction_buffer.append(prediction)

        avg_pred = np.mean(prediction_buffer, axis=0)
        class_id = np.argmax(avg_pred)
        confidence = np.max(avg_pred)

        letter = actions[class_id]

        # Display
        cv2.rectangle(frame, (0, 0), (320, 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"LETTER: {letter}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3
        )
        cv2.putText(
            frame,
            f"CONF: {confidence:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "SHOW ONE HAND",
            (60, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
        current_time = time.time()

        # --- NEW: Stability Logic ---
        if confidence > CONFIDENCE_THRESHOLD:
            if letter == current_stable_letter:
                stable_frames += 1
            else:
                current_stable_letter = letter
                stable_frames = 1

            if stable_frames == REQUIRED_FRAMES:
                if len(letter_buffer) == 0 or letter != letter_buffer[-1]:
                    letter_buffer.append(letter)
                    last_seen_time = current_time
        else:
            # Reset stable frames if confidence drops
            stable_frames = 0
            current_stable_letter = ""

        # Display letter
        cv2.rectangle(frame, (0, 0), (450, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"LETTER: {letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"CONF: {confidence:.2f} | STABLE: {stable_frames}/{REQUIRED_FRAMES}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # =============================================
    # NO HAND → FINALIZE WORD
    # =============================================
    else:
        # Reset counters when hand leaves screen
        stable_frames = 0
        current_stable_letter = ""

        if len(letter_buffer) > 0 and (time.time() - last_seen_time) > PAUSE_TIME:
            word = "".join(letter_buffer)
            print("Detected Word:", word)

            letter_buffer = []

        cv2.putText(frame, "SHOW ONE HAND", (60, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # =============================================
    # DISPLAY BUFFER + WORD
    # =============================================
    cv2.putText(frame, f"Buffer: {''.join(letter_buffer)}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Word: {word}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
