"""
preprocess_words.py
====================
Extract hand + upper-body pose landmarks from every MP4 in Labeled/
and save one .npy file per video into My_Word_Data/<WORD>/<id>.npy.

Feature layout per frame (150 dims):
  [0 :63]  → left  hand:  21 landmarks × (x,y,z) wrist-relative
  [63:126] → right hand:  21 landmarks × (x,y,z) wrist-relative
  [126:150]→ pose  body:   8 landmarks × (x,y,z)   (both shoulders,
                           elbows, wrists, hips pinned so motion shows)

Sequence length: 30 frames (evenly sampled from each video).
Output shape:    (30, 150)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================================================
# 1. PATHS
# ==================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LABELED_PATH = os.path.join(SCRIPT_DIR, "Labeled")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "My_Word_Data")
HAND_MODEL   = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
POSE_MODEL   = os.path.join(SCRIPT_DIR, "pose_landmarker.task")

N_FRAMES   = 30          # frames to sample per video
FRAME_DIM  = 150         # dims per frame (63+63+24)
HAND_DIM   = 63          # 21 × 3
POSE_IDXS  = [11,12,13,14,15,16,23,24]  # shoulders, elbows, wrists, hips

for path in (HAND_MODEL, POSE_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model: {path}\n"
            "Run: python -c \"import urllib.request; "
            "urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',"
            " 'pose_landmarker.task')\""
        )

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==================================================
# 2. BUILD MEDIAPIPE DETECTORS
# ==================================================
hand_opts = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

pose_opts = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

print("✅ MediaPipe detectors ready.")


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================
def hand_landmarks_to_vec(hand_list, handedness_list, n_hands=2):
    """
    Map detected hands to fixed left/right slots → shape (2, 63).
    Missing hand → zero vector.
    """
    slots = {0: np.zeros(HAND_DIM, dtype=np.float32),   # LEFT
             1: np.zeros(HAND_DIM, dtype=np.float32)}   # RIGHT

    for hand, cat in zip(hand_list, handedness_list):
        side = 0 if cat[0].category_name == "Left" else 1
        pts  = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        pts -= pts[0]                        # wrist-relative
        slots[side] = pts.flatten()

    return np.concatenate([slots[0], slots[1]])   # (126,)


def pose_landmarks_to_vec(pose_list):
    """Extract 8 upper-body pose landmarks → shape (24,)."""
    if not pose_list:
        return np.zeros(len(POSE_IDXS) * 3, dtype=np.float32)

    lms = pose_list[0]
    pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in POSE_IDXS],
                   dtype=np.float32)
    # Make pose wrist-relative (use mid-shoulder as anchor)
    anchor = (pts[0] + pts[1]) / 2.0
    pts   -= anchor
    return pts.flatten()    # (24,)


def sample_frames(cap, n=N_FRAMES):
    """
    Read all frames from an open VideoCapture and return n evenly-spaced ones.
    Returns list of BGR numpy arrays.
    """
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    if len(all_frames) == 0:
        return []

    if len(all_frames) <= n:
        # Pad by repeating last frame
        idxs = list(range(len(all_frames)))
        while len(idxs) < n:
            idxs.append(idxs[-1])
    else:
        idxs = np.linspace(0, len(all_frames) - 1, n, dtype=int).tolist()

    return [all_frames[i] for i in idxs]


def extract_sequence(video_path):
    """Process one MP4 → numpy array of shape (N_FRAMES, FRAME_DIM)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  Cannot open: {video_path}")
        return None

    frames = sample_frames(cap)
    cap.release()

    if not frames:
        return None

    sequence = []
    for bgr in frames:
        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        hand_vec = hand_landmarks_to_vec(
            hand_result.hand_landmarks,
            hand_result.handedness
        )
        pose_vec = pose_landmarks_to_vec(pose_result.pose_landmarks)

        frame_vec = np.concatenate([hand_vec, pose_vec])   # (150,)
        sequence.append(frame_vec)

    return np.array(sequence, dtype=np.float32)   # (N_FRAMES, FRAME_DIM)


# ==================================================
# 4. PROCESS ALL WORDS
# ==================================================
total_saved = 0
skipped     = 0

words = sorted([
    d for d in os.listdir(LABELED_PATH)
    if os.path.isdir(os.path.join(LABELED_PATH, d))
])

print(f"\n📂 Found {len(words)} word classes: {words}\n")

for word in words:
    word_in  = os.path.join(LABELED_PATH, word)
    word_out = os.path.join(OUTPUT_PATH, word)
    os.makedirs(word_out, exist_ok=True)

    mp4_files = sorted([
        f for f in os.listdir(word_in)
        if f.lower().endswith(".mp4")
    ])

    print(f"▶  {word} ({len(mp4_files)} videos)")

    for mp4 in mp4_files:
        video_path = os.path.join(word_in, mp4)
        out_path   = os.path.join(word_out, mp4.replace(".mp4", ".npy"))

        seq = extract_sequence(video_path)
        if seq is None or seq.shape != (N_FRAMES, FRAME_DIM):
            print(f"    ❌ Skipped (bad shape): {mp4}")
            skipped += 1
            continue

        np.save(out_path, seq)
        total_saved += 1
        print(f"    ✅ {mp4}  →  {seq.shape}")

print(f"\n✅ Done. Saved {total_saved} sequences, skipped {skipped}.")
print(f"   Output: {OUTPUT_PATH}")
