import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==================================================
# 1. LOAD DATA FROM 'My_Keypoint_Data'
# ==================================================
DATA_PATH = os.path.join(os.getcwd(), "My_Keypoint_Data")

actions = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and len(d) == 1 and d.isalpha()
])

actions = np.array(actions)
print("Detected Classes:", actions)

X_raw, y_raw = [], []
label_map = {label: idx for idx, label in enumerate(actions)}

print(f"📂 Loading .npy files from {DATA_PATH}...")

for letter in actions:
    letter_path = os.path.join(DATA_PATH, letter)
    for file in os.listdir(letter_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(letter_path, file))
            if data.shape == (63,):
                # Convert to wrist-relative coordinates
                points = data.reshape(21, 3)
                wrist = points[0]
                relative_points = points - wrist
                X_raw.append(relative_points.flatten())
                y_raw.append(label_map[letter])

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw)
print(f"✅ Loaded {X_raw.shape[0]} original samples across {len(actions)} classes.")


# ==================================================
# 2. DATA AUGMENTATION  (applied BEFORE scaler fit)
# ==================================================
def augment_sample(points_flat, n_augments=5):
    """
    Generate n_augments variants of a 63-dim wrist-relative hand sample.
    Augmentations applied:
      • Gaussian jitter          – simulates webcam noise
      • Random XY scale          – simulates distance changes
      • Random Z-axis scale      – simulates depth variation
      • Random small rotation    – simulates hand tilt/twist
      • Random per-finger swap   – left↔right mirror
    """
    pts = points_flat.reshape(21, 3)
    augmented = []

    for _ in range(n_augments):
        aug = pts.copy()

        # 1. Gaussian jitter (1 % of hand scale)
        aug += np.random.normal(0, 0.01, aug.shape)

        # 2. Random XY scale [0.85 – 1.15]
        xy_scale = np.random.uniform(0.85, 1.15)
        aug[:, :2] *= xy_scale

        # 3. Random Z scale [0.80 – 1.20]
        z_scale = np.random.uniform(0.80, 1.20)
        aug[:, 2] *= z_scale

        # 4. Random in-plane rotation [±10 °]
        angle = np.random.uniform(-10, 10) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated_x = aug[:, 0] * cos_a - aug[:, 1] * sin_a
        rotated_y = aug[:, 0] * sin_a + aug[:, 1] * cos_a
        aug[:, 0] = rotated_x
        aug[:, 1] = rotated_y

        # 5. Random horizontal mirror (50 % chance)
        if np.random.rand() < 0.5:
            aug[:, 0] = -aug[:, 0]

        augmented.append(aug.flatten())

    return augmented


print("🔄 Augmenting data...")
X_aug, y_aug = list(X_raw), list(y_raw)

for sample, label in zip(X_raw, y_raw):
    for aug_sample in augment_sample(sample, n_augments=5):
        X_aug.append(aug_sample)
        y_aug.append(label)

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug)
print(f"✅ Augmented dataset: {X_aug.shape[0]} total samples.")


# ==================================================
# 3. TRAIN / TEST SPLIT  (split BEFORE fitting scaler)
# ==================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
)


# ==================================================
# 4. STANDARDISATION  (fit on train only, save for inference)
# ==================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Save the scaler so gui_app / realtime_test can use the same normalisation
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")

y_train_cat = to_categorical(y_train, num_classes=len(actions))
y_test_cat  = to_categorical(y_test,  num_classes=len(actions))


# ==================================================
# 5. CLASS WEIGHTS  (handle imbalanced letter counts)
# ==================================================
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
print("⚖️  Class weights computed (balanced).")


# ==================================================
# 6. DNN MODEL  — architecture UNCHANGED
# ==================================================
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(len(actions), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# ==================================================
# 7. CALLBACKS
# ==================================================
callbacks = [
    # Stop early when val_accuracy stops improving
    EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Halve the learning rate when val_loss plateaus for 7 epochs
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    # Always save the epoch with the best validation accuracy
    ModelCheckpoint(
        filepath="sign_language_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
]


# ==================================================
# 8. TRAIN
# ==================================================
print("🚀 Training started...")
history = model.fit(
    X_train, y_train_cat,
    epochs=150,           # EarlyStopping will terminate before 150 if appropriate
    batch_size=64,        # Larger batch → more stable gradient estimates
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)
print("✅ Best model saved as sign_language_model.keras")


# ==================================================
# 9. TEST SET PERFORMANCE
# ==================================================
y_pred       = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true        = y_test  # already integer labels

print("\n📊 Classification Report (Test Set):")
print(classification_report(y_true, y_pred_labels, target_names=actions))


# ==================================================
# 10. CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_true, y_pred_labels)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    xticklabels=actions,
    yticklabels=actions,
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# ==================================================
# 11. TRAINING CURVES
# ==================================================
plt.figure()
plt.plot(history.history["accuracy"],     label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()

plt.figure()
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()


# ==================================================
# 12. FINAL TEST ACCURACY
# ==================================================
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"🎯 Final Test Accuracy: {acc * 100:.2f}%")