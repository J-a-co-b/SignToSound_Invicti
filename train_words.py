"""
train_words.py  —  Efficient, accurate word-level 1D-CNN trainer
================================================================
Designed to run at full inference speed on CPU-only hardware
(Raspberry Pi 3B+, laptops, desktops without GPU).

Architecture: Depthwise-Separable 1D-CNN
  • Minimal parameters (~15K) → fast inference on Pi
  • High accuracy via aggressive augmentation and label smoothing
  • Outputs TF-Lite .tflite model alongside .keras for Pi deployment

Input:  My_Word_Data/<WORD>/<id>.npy   shape (30, 150)
Output: word_model.keras               Keras model (desktop use)
        word_model.tflite              TFLite model  (Pi use)
        word_scaler.pkl                StandardScaler
        word_label_map.json            {index: word}
        word_confusion_matrix.png
        word_accuracy_curve.png / word_loss_curve.png
"""

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.metrics          import classification_report, confusion_matrix
from sklearn.preprocessing    import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, DepthwiseConv1D, SeparableConv1D,
    BatchNormalization, Dropout, GlobalAveragePooling1D,
    Dense, Add, Activation, Multiply, Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ── Focal Loss (boosts hard-to-classify similar words) ────────────────────
def focal_loss(gamma=2.0):
    """Focal loss — down-weights easy samples, forces model to learn hard ones."""
    def _loss(y_true, y_pred):
        import tensorflow as tf
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return _loss

# ======================================================
# 1. CONSTANTS
# ======================================================
DATA_PATH  = os.path.join(os.getcwd(), "My_Word_Data")
N_FRAMES   = 30
FRAME_DIM  = 150
BATCH_SIZE = 32      # larger batch = more stable gradients with bigger dataset
MAX_EPOCHS = 300
AUG_PER_SAMPLE = 8  # augmented copies per original sample
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ======================================================
# 2. LOAD DATA
# ======================================================
words = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])
label_map = {label: idx for idx, label in enumerate(words)}
n_classes  = len(words)
print(f"📂 Found {n_classes} classes: {words}\n")

X_raw, y_raw = [], []
for word in words:
    word_path = os.path.join(DATA_PATH, word)
    for f in sorted(os.listdir(word_path)):
        if not f.endswith(".npy"):
            continue
        seq = np.load(os.path.join(word_path, f))
        if seq.shape != (N_FRAMES, FRAME_DIM):
            print(f"  ⚠️  Bad shape {seq.shape}: {f} — skipping")
            continue
        X_raw.append(seq)
        y_raw.append(label_map[word])

X_raw = np.array(X_raw, dtype=np.float32)   # (N, 30, 150)
y_raw = np.array(y_raw)
print(f"✅ Loaded {X_raw.shape[0]} sequences across {n_classes} classes.")
for w in words:
    cnt = np.sum(y_raw == label_map[w])
    print(f"   {w:>12}: {cnt} samples")


# ======================================================
# 3. AUGMENTATION  (match training distribution to real-time)
# ======================================================
def augment_sequence(seq, n=AUG_PER_SAMPLE):
    """
    Return n augmented variants of a (T, D) landmark sequence.
    All augmentations preserve the wrist-relative, unit-scaled structure
    used in both preprocess_words.py and the live GUI extraction.
    """
    T, D = seq.shape
    augmented = []
    for _ in range(n):
        s = seq.copy()

        # 1. Gaussian noise (simulates natural hand tremor)
        noise_scale = np.random.uniform(0.004, 0.015)
        s += np.random.normal(0, noise_scale, s.shape)

        # 2. Time reverse (50 % chance)  — valid for many symmetric signs
        if np.random.rand() < 0.5:
            s = s[::-1].copy()

        # 3. Speed jitter ±25 %
        speed_factor = np.random.uniform(0.75, 1.25)
        new_len      = max(3, int(T * speed_factor))
        src_idx      = np.linspace(0, T - 1, new_len)
        dst_idx      = np.linspace(0, new_len - 1, T)
        s_speed      = np.zeros_like(s)
        for d in range(D):
            s_speed[:, d] = np.interp(dst_idx, np.arange(new_len),
                                      np.interp(src_idx, np.arange(T), s[:, d]))
        s = s_speed

        # 4. Spatial scale ±15 % on x/y (every 3rd pair of dims)
        xy_scale = np.random.uniform(0.85, 1.15)
        for start in range(0, D, 3):
            if start + 1 < D:
                s[:, start]     *= xy_scale
                s[:, start + 1] *= xy_scale

        # 5. Frame dropout: randomly zero out 1-2 frames (simulates occlusion)
        n_drop = np.random.randint(0, 3)
        drop_idx = np.random.choice(T, n_drop, replace=False)
        s[drop_idx] = 0.0

        # 6. Perspective jitter — shear XY relative to Z depth
        shear = np.random.uniform(-0.04, 0.04)
        for start in range(0, D, 3):
            if start + 2 < D:
                s[:, start]     += shear * s[:, start + 2]
                s[:, start + 1] += shear * s[:, start + 2]

        # 7. Landmark dropout — randomly zero 1-3 landmark dimensions
        if np.random.rand() < 0.3:
            drop_dims = np.random.choice(D, np.random.randint(1, 4), replace=False)
            s[:, drop_dims] = 0.0

        augmented.append(s)
    return augmented


print(f"\n🔄 Augmenting (×{AUG_PER_SAMPLE} per sample) …")
X_aug, y_aug = list(X_raw), list(y_raw)
for seq, label in zip(X_raw, y_raw):
    for aug_seq in augment_sequence(seq):
        X_aug.append(aug_seq)
        y_aug.append(label)

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug)
print(f"✅ Augmented dataset: {X_aug.shape[0]} sequences  "
      f"({X_raw.shape[0]} original + "
      f"{X_aug.shape[0]-X_raw.shape[0]} synthetic)\n")


# ======================================================
# 4. TRAIN / TEST SPLIT
# ======================================================
X_train_r, X_test_r, y_train, y_test = train_test_split(
    X_aug, y_aug,
    test_size=0.15,
    stratify=y_aug,
    random_state=RANDOM_SEED
)

# ======================================================
# 5. NORMALISE  — per-feature StandardScaler (fit on train only)
# ======================================================
N_tr = X_train_r.shape[0]
N_te = X_test_r.shape[0]

X_train_flat = X_train_r.reshape(N_tr, -1)   # (N, 4500)
X_test_flat  = X_test_r.reshape(N_te, -1)

scaler        = StandardScaler()
X_train_flat  = scaler.fit_transform(X_train_flat)
X_test_flat   = scaler.transform(X_test_flat)

X_train = X_train_flat.reshape(N_tr, N_FRAMES, FRAME_DIM)
X_test  = X_test_flat.reshape(N_te,  N_FRAMES, FRAME_DIM)

joblib.dump(scaler, "word_scaler.pkl")
print("✅ word_scaler.pkl saved.\n")

y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

# ======================================================
# 6. CLASS WEIGHTS  (handle minor imbalance)
# ======================================================
cw_arr  = compute_class_weight("balanced",
                               classes=np.unique(y_train),
                               y=y_train)
cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}
print("⚖️  Class weights (balanced):\n",
      {words[k]: round(v, 3) for k, v in cw_dict.items()}, "\n")

# ======================================================
# 7. MODEL — Lightweight Separable 1D-CNN
# ======================================================
#   Designed for CPU inference: SeparableConv1D splits depthwise +
#   pointwise convolutions, giving ~8× fewer multiply-adds vs standard
#   Conv1D with the same filter count.  Total params ≈ 30K.
# ======================================================
def build_model(n_classes, n_frames=N_FRAMES, frame_dim=FRAME_DIM):
    inp = Input(shape=(n_frames, frame_dim), name="landmarks")

    # Block 1 — local feature extraction
    x = SeparableConv1D(64, kernel_size=3, padding="same",
                        depthwise_regularizer=l2(1e-4),
                        pointwise_regularizer=l2(1e-4),
                        name="sep_conv1")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    # Block 2 — temporal context
    x = SeparableConv1D(64, kernel_size=5, padding="same",
                        depthwise_regularizer=l2(1e-4),
                        pointwise_regularizer=l2(1e-4),
                        name="sep_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    # Block 3 — wider context
    x = SeparableConv1D(32, kernel_size=7, padding="same",
                        depthwise_regularizer=l2(1e-4),
                        pointwise_regularizer=l2(1e-4),
                        name="sep_conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.20)(x)

    # Global aggregation
    x = GlobalAveragePooling1D()(x)

    # Classifier head
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.25)(x)
    out = Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="SignToSound_Word")
    return model


model = build_model(n_classes)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0),
    metrics=["accuracy"]
)
model.summary()
total_params = model.count_params()
print(f"\n📐 Total parameters: {total_params:,}  "
      f"({'✅ Pi-safe (<100K)' if total_params < 100_000 else '⚠️ Consider reducing'})\n")

# Save label map
label_index_to_word = {str(v): k for k, v in label_map.items()}
with open("word_label_map.json", "w") as f:
    json.dump(label_index_to_word, f, indent=2)
print("✅ word_label_map.json saved.\n")


# ======================================================
# 8. CALLBACKS
# ======================================================
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="word_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
]


# ======================================================
# 9. TRAIN
# ======================================================
print("🚀 Training …\n")
history = model.fit(
    X_train, y_train_cat,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.12,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1,
)
print("\n✅ Best model saved → word_model.keras\n")


# ======================================================
# 10. EVALUATE
# ======================================================
y_pred_probs  = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=words))

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🎯 Final Test Accuracy : {acc * 100:.2f}%")
print(f"   Final Test Loss     : {loss:.4f}\n")


# ======================================================
# 11. CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=words, yticklabels=words,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Word Model — Confusion Matrix")
plt.tight_layout()
plt.savefig("word_confusion_matrix.png", dpi=120)
print("✅ word_confusion_matrix.png saved.")


# ======================================================
# 12. TRAINING CURVES
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("word_accuracy_curve.png", dpi=120)
print("✅ word_accuracy_curve.png saved.")


# ======================================================
# 13. EXPORT TO TFLITE  (for Raspberry Pi deployment)
# ======================================================
print("\n🔄 Converting to TFLite (Pi-optimised) …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]    # dynamic range quant
tflite_model = converter.convert()
with open("word_model.tflite", "wb") as f:
    f.write(tflite_model)
tflite_size_kb = len(tflite_model) / 1024
print(f"✅ word_model.tflite saved  ({tflite_size_kb:.1f} KB)\n")

print("=" * 55)
print("  All done!  Files ready:")
print("    word_model.keras   — full Keras (desktop GUI)")
print("    word_model.tflite  — quantised  (Raspberry Pi)")
print("    word_scaler.pkl    — feature scaler")
print("    word_label_map.json— label map")
print("=" * 55)
