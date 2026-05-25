import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================================================
# 1. DATASET PATHS (MATCHES YOUR FOLDER STRUCTURE)
# ==================================================
BASE_PATH = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "American Sign Language Dataset",
    "American Sign Language Dataset",
    "Pre_Processed Data"
    #"Augmented Data"
)

#TRAIN_FOLDERS = ["Train Data 1", "Train Data 2"]
TRAIN_FOLDERS = ["Histogram Data 1"]
OUTPUT_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==================================================
# 2. MEDIAPIPE HAND LANDMARKER
# ==================================================
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

print("🚀 Extracting landmarks from Augmented Train Data...")

# ==================================================
# 3. PROCESS TRAIN DATA 1 & 2
# ==================================================
for train_folder in TRAIN_FOLDERS:
    train_path = os.path.join(BASE_PATH, train_folder)
    if not os.path.exists(train_path):
        continue

    print(f"\n📂 Processing {train_folder}")

    for letter in sorted(os.listdir(train_path)):
        letter_path = os.path.join(train_path, letter)

        # Only A–Z folders
        if not os.path.isdir(letter_path):
            continue
        if len(letter) != 1 or not letter.isalpha():
            continue

        out_dir = os.path.join(OUTPUT_PATH, letter)
        os.makedirs(out_dir, exist_ok=True)

        count = len(os.listdir(out_dir))  # continue numbering

        for img_name in os.listdir(letter_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(letter_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)
            if not result.hand_landmarks:
                continue

            hand = result.hand_landmarks[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand],
                dtype=np.float32
            ).flatten()

            np.save(os.path.join(out_dir, f"{count}.npy"), landmarks)
            count += 1

        print(f"  {letter}: {count} samples")

print("\n✅ My_Keypoint_Data successfully generated.")
