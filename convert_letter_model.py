import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

print("Rebuilding Letter Model Architecture...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    BatchNormalization(momentum=0.99, epsilon=0.001),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(momentum=0.99, epsilon=0.001),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(24, activation='softmax'),
])

print("Loading weights from sign_language_model.weights.h5...")
model.load_weights('sign_language_model.weights.h5')

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done! sign_language_model.tflite saved successfully.")
print(f"File size: {len(tflite_model) / 1024:.1f} KB")
