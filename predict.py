# predict.py -- Quick image test using your trained CNN
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

MODEL_PATH = "./tmp_checkpoint/deepfake_cnn_model.h5"
IMG_SIZE = (128, 128)

if len(sys.argv) < 2:
    print("❌ Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
    sys.exit(1)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = np.expand_dims(img, axis=0) / 255.0

# Predict
pred = model.predict(img)[0][0]
label = "FAKE" if pred < 0.5 else "REAL"
confidence = (1 - pred if label == "FAKE" else pred) * 100

print(f"\n🧠 Prediction: {label}")
print(f"🎯 Confidence: {confidence:.2f}%")
