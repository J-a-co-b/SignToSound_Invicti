import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, SeparableConv1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_FRAMES = 30
FEATURES = 150

def build_model(n_classes):
    inp = Input(shape=(WORD_FRAMES, FEATURES), name="landmarks")
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
    
    model = Model(inp, out, name="SignToSound_Word")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    data_path = os.path.join(SCRIPT_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run preprocess_dataset.py first.")
        return

    print("Loading preprocessed data...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    # Load label map to get number of classes
    with open(os.path.join(SCRIPT_DIR, "word_label_map.json"), "r") as f:
        label_map = json.load(f)
    n_classes = len(label_map)
    
    print(f"Data shape: {X.shape}, labels shape: {y.shape}")
    print(f"Number of classes: {n_classes}")
    
    # Convert labels to categorical
    y_cat = to_categorical(y, num_classes=n_classes)
    
    # Flatten X to scale, then reshape back
    print("Scaling features...")
    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X = X_scaled_flat.reshape(X.shape[0], WORD_FRAMES, FEATURES)
    
    # Save the scaler
    scaler_path = os.path.join(SCRIPT_DIR, "word_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    model = build_model(n_classes)
    model.summary()
    
    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(SCRIPT_DIR, "word_model.keras"),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor='val_accuracy'
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    # Save the full model and weights separately
    model.save(os.path.join(SCRIPT_DIR, "word_model.keras"))
    model.save_weights(os.path.join(SCRIPT_DIR, "word_model.weights.h5"))
    print("Saved Keras model and weights.")
    
    # Convert to TFLite
    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(os.path.join(SCRIPT_DIR, "word_model.tflite"), "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model.")
    print("Training process complete!")

if __name__ == "__main__":
    main()
