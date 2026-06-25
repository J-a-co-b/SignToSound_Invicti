import os
import glob
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

DATASET_DIR_OLD = r"D:\archive\dataset\SL"
DATASET_DIR_NEW = r"D:\archive\dataset_new"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Original 44 words
ORIGINAL_WORDS = [
    "DRINK", "EAT", "EMERGENCY", "HELLO", "HELP", "HOSPITAL", "MEDICINE", 
    "MORE", "NO", "PAIN", "PLEASE", "SICK", "THANK YOU", "WANT", "YES", 
    "college", "doctor", "me", "meet", "on", "parents", "satisfy", "their", 
    "them", "then", "they", "visit", "wait", "war", "way", "we", "wear", 
    "week", "wheelchair", "where", "which", "who", "why", "without", 
    "witness", "wow", "you", "your", "yourself"
]

# New 26 words requested
NEW_WORDS = [
    "brother", "mother", "father", "family", "boy", "girl", "man", "go", 
    "see", "know", "feel", "like", "love", "can", "good", "bad", "happy", 
    "deaf", "how", "name", "and", "but", "because", "day", "all", "house"
]

ALL_WORDS = ORIGINAL_WORDS + NEW_WORDS
WORD_FRAMES = 30
POSE_IDXS = [11, 12, 13, 14, 15, 16, 23, 24]

def main():
    print("Setting up MediaPipe models...")
    hand_model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
    pose_model_path = os.path.join(SCRIPT_DIR, "pose_landmarker.task")

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4
    )

    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    def extract_frame_features(mp_image):
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)
        
        slots = {0: np.zeros(63, dtype=np.float32), 1: np.zeros(63, dtype=np.float32)}
        if hand_result.hand_landmarks:
            for hand, cat in zip(hand_result.hand_landmarks, hand_result.handedness):
                side = 0 if cat[0].category_name == "Left" else 1
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                pts -= pts[0]
                slots[side] = pts.flatten()
                
        pose_vec = np.zeros(24, dtype=np.float32)
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in POSE_IDXS], dtype=np.float32)
            pose_vec = (pts - (pts[0] + pts[1]) / 2.0).flatten()
            
        return np.concatenate([slots[0], slots[1], pose_vec])

    X, y = [], []
    
    # Save the new word map
    word_label_map = {str(i): word for i, word in enumerate(ALL_WORDS)}
    with open(os.path.join(SCRIPT_DIR, "word_label_map.json"), "w") as f:
        json.dump(word_label_map, f, indent=2)
        
    for label_idx, word in enumerate(ALL_WORDS):
        folder_name = word.lower()
        
        # Determine which folder to use based on whether it's an old or new word
        if word in ORIGINAL_WORDS:
            folder_path = os.path.join(DATASET_DIR_OLD, folder_name)
        else:
            folder_path = os.path.join(DATASET_DIR_NEW, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for word '{word}' at {folder_path}")
            continue
            
        video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                      glob.glob(os.path.join(folder_path, "*.mkv")) + \
                      glob.glob(os.path.join(folder_path, "*.avi"))
                      
        print(f"Processing '{word}' ({len(video_files)} videos)...")
        
        for video_file in tqdm(video_files, leave=False):
            cap = cv2.VideoCapture(video_file)
            frames_features = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                feats = extract_frame_features(mp_image)
                frames_features.append(feats)
                
            cap.release()
            
            if frames_features:
                raw = np.array(frames_features)
                if len(raw) > 0:
                    idxs = np.linspace(0, len(raw) - 1, WORD_FRAMES, dtype=int)
                    seq = raw[idxs]
                    X.append(seq)
                    y.append(label_idx)
                
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nExtraction complete. Total samples: {len(X)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    np.savez(os.path.join(SCRIPT_DIR, "processed_data.npz"), X=X, y=y)
    print("Saved to processed_data.npz")

if __name__ == "__main__":
    main()
