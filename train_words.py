"""
train_words.py
==============
Train a 1-D CNN sequence model on the word-level landmark data
produced by preprocess_words.py.

Input:  My_Word_Data/<WORD>/<id>.npy  — shape (30, 150)
Output: word_model.keras              — Keras model
        word_scaler.pkl               — StandardScaler (per-feature)
        word_label_map.json           — {index: word} mapping
        word_confusion_matrix.png
        word_accuracy_curve.png
        word_loss_curve.png
"""

import os
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Dropout,
    GlobalAveragePooling1D, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. CONSTANTS
# ==================================================
DATA_PATH = os.path.join(os.getcwd(), "My_Word_Data")
N_FRAMES  = 30
FRAME_DIM = 150


# ==================================================
# 2. LOAD DATA
# ==================================================
words = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])
label_map = {label: idx for idx, label in enumerate(words)}
print("Classes:", words)

X_raw, y_raw = [], []

for word in words:
    word_path = os.path.join(DATA_PATH, word)
    for f in sorted(os.listdir(word_path)):
        if not f.endswith(".npy"):
            continue
        seq = np.load(os.path.join(word_path, f))
        if seq.shape != (N_FRAMES, FRAME_DIM):
            print(f"  ⚠️  Skipping bad shape {seq.shape}: {f}")
            continue
        X_raw.append(seq)
        y_raw.append(label_map[word])

X_raw = np.array(X_raw, dtype=np.float32)   # (N, 30, 150)
y_raw = np.array(y_raw)
print(f"✅ Loaded {X_raw.shape[0]} sequences across {len(words)} classes.")


# ==================================================
# 3. AUGMENTATION
# ==================================================
def augment_sequence(seq, n=6):
    """
    Return n augmented variants of a (T, D) sequence.
    Augmentations:
      • Gaussian noise  – random jitter on all coordinates
      • Time reverse    – plays the sign backwards
      • Speed jitter    – stretch/compress time axis by ±20 %
      • Spatial scale   – scale x/y by ±10 %
    """
    T, D = seq.shape
    augmented = []

    for _ in range(n):
        s = seq.copy()

        # 1. Gaussian noise
        s += np.random.normal(0, 0.008, s.shape)

        # 2. Time reverse (50 % chance)
        if np.random.rand() < 0.5:
            s = s[::-1].copy()

        # 3. Speed jitter: resample to random length then re-sample back to T
        speed_factor = np.random.uniform(0.8, 1.2)
        new_len = max(2, int(T * speed_factor))
        src_indices = np.linspace(0, T - 1, new_len)
        dst_indices = np.linspace(0, new_len - 1, T)
        s_speed = np.zeros_like(s)
        for d in range(D):
            s_speed[:, d] = np.interp(dst_indices, np.arange(new_len),
                                      np.interp(src_indices,
                                                np.arange(T), s[:, d]))
        s = s_speed

        # 4. Spatial scale on x/y channels (every 3rd dim offset 0 and 1)
        xy_scale = np.random.uniform(0.9, 1.1)
        for start in range(0, D, 3):
            s[:, start]     *= xy_scale   # x
            s[:, start + 1] *= xy_scale   # y

        augmented.append(s)

    return augmented


print("🔄 Augmenting sequences ...")
X_aug, y_aug = list(X_raw), list(y_raw)

for seq, label in zip(X_raw, y_raw):
    for aug_seq in augment_sequence(seq, n=6):
        X_aug.append(aug_seq)
        y_aug.append(label)

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug)
print(f"✅ Augmented dataset: {X_aug.shape[0]} sequences.")


# ==================================================
# 4. SPLIT
# ==================================================
X_train_r, X_test_r, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)


# ==================================================
# 5. NORMALISE  (per-feature StandardScaler, fit on train)
# ==================================================
# Flatten T×D → (N, T*D) for scaler, then reshape back
N_tr = X_train_r.shape[0]
N_te = X_test_r.shape[0]

X_train_flat = X_train_r.reshape(N_tr, -1)
X_test_flat  = X_test_r.reshape(N_te, -1)

scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat  = scaler.transform(X_test_flat)

X_train = X_train_flat.reshape(N_tr, N_FRAMES, FRAME_DIM)
X_test  = X_test_flat.reshape(N_te, N_FRAMES, FRAME_DIM)

joblib.dump(scaler, "word_scaler.pkl")
print("✅ Word scaler saved as word_scaler.pkl")

y_train_cat = to_categorical(y_train, num_classes=len(words))
y_test_cat  = to_categorical(y_test,  num_classes=len(words))


# ==================================================
# 6. CLASS WEIGHTS
# ==================================================
cw_arr  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = {i: w for i, w in enumerate(cw_arr)}
print("⚖️  Class weights (balanced).")


# ==================================================
# 7. MODEL  — 1-D CNN sequence classifier
# ==================================================
model = Sequential([
    Conv1D(128, kernel_size=3, padding="same", activation="relu",
           input_shape=(N_FRAMES, FRAME_DIM)),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(64, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    GlobalAveragePooling1D(),

    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(words), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Save label map
label_index_to_word = {str(v): k for k, v in label_map.items()}
with open("word_label_map.json", "w") as f:
    json.dump(label_index_to_word, f, indent=2)
print("✅ word_label_map.json saved.")


# ==================================================
# 8. CALLBACKS
# ==================================================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath="word_model.keras",
                    monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]


# ==================================================
# 9. TRAIN
# ==================================================
print("🚀 Training word model ...")
history = model.fit(
    X_train, y_train_cat,
    epochs=200,
    batch_size=16,        # small batch — small dataset
    validation_split=0.1,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1,
)
print("✅ Best word model saved as word_model.keras")


# ==================================================
# 10. EVALUATE
# ==================================================
y_pred        = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=words))


# ==================================================
# 11. CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, xticklabels=words, yticklabels=words,
            cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Word Model — Confusion Matrix")
plt.tight_layout()
plt.savefig("word_confusion_matrix.png")
plt.show()


# ==================================================
# 12. CURVES
# ==================================================
plt.figure()
plt.plot(history.history["accuracy"],     label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Word Model — Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("word_accuracy_curve.png")
plt.show()

plt.figure()
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Word Model — Loss")
plt.legend(); plt.tight_layout()
plt.savefig("word_loss_curve.png")
plt.show()


# ==================================================
# 13. FINAL TEST ACCURACY
# ==================================================
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"🎯 Final Word Model Test Accuracy: {acc * 100:.2f}%")
