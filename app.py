import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

MODEL_PATH = "./tmp_checkpoint/deepfake_cnn_model.h5"
REAL_DIR = "./custom_dataset/real"
FAKE_DIR = "./custom_dataset/fake"
IMG_SIZE = (128, 128)

# ======================
# Load model once
# ======================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ======================
# Helper functions
# ======================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return np.expand_dims(img, axis=0) / 255.0

def predict_image(img_array):
    pred = model.predict(img_array)[0][0]
    label = "REAL" if pred > 0.5 else "FAKE"
    confidence = (pred if pred > 0.5 else 1 - pred) * 100
    return label, confidence

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="wide")
st.title("🧠 Deepfake Image Detector")

tab1, tab2 = st.tabs(["📁 Scan Dataset Folders", "📤 Upload & Test Single Image"])

# ======================================================
# TAB 1 — Dataset Folder Scan
# ======================================================
with tab1:
    st.subheader("📂 Scan and Evaluate Real vs Fake Folders")

    if st.button("🔍 Start Scanning"):
        y_true = []
        y_pred = []
        details = []

        total_images = 0

        for category, folder in [("REAL", REAL_DIR), ("FAKE", FAKE_DIR)]:
            if not os.path.exists(folder):
                st.error(f"❌ Folder not found: {folder}")
                continue

            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(image_files)

            with st.spinner(f"Scanning {category} folder..."):
                for img_name in image_files:
                    path = os.path.join(folder, img_name)
                    arr = preprocess_image(path)
                    pred_label, conf = predict_image(arr)
                    y_true.append(category)
                    y_pred.append(pred_label)
                    details.append((img_name, category, pred_label, conf, path))

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred) * 100
            cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])

            st.success(f"✅ Scanned {total_images} images successfully!")
            st.markdown(f"### 🎯 **Overall Accuracy:** `{acc:.2f}%`")

            st.write("#### 📊 Confusion Matrix")
            st.write(f"REAL correctly identified: {cm[0][0]}")
            st.write(f"FAKE correctly identified: {cm[1][1]}")
            st.write(f"Mistaken REAL→FAKE: {cm[0][1]}")
            st.write(f"Mistaken FAKE→REAL: {cm[1][0]}")

            st.write("#### 🧾 Sample Predictions with Images")
            cols = st.columns(3)
            for i, (name, true, pred, conf, path) in enumerate(details[:15]):  # show first 15 with images
                with cols[i % 3]:
                    img = Image.open(path)
                    st.image(img, caption=f"{name}\nTrue: {true}\nPred: {pred} ({conf:.1f}%)", use_container_width=True)
        else:
            st.warning("⚠️ No images found in your folders.")

# ======================================================
# TAB 2 — Manual Upload
# ======================================================
with tab2:
    st.subheader("📤 Upload a Single Image for Prediction")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Predict
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_input = np.expand_dims(img_resized, axis=0) / 255.0
        pred = model.predict(img_input)[0][0]
        label = "🟢 REAL" if pred > 0.5 else "🔴 FAKE"
        confidence = (pred if pred > 0.5 else 1 - pred) * 100

        # Display result side-by-side
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: **{confidence:.2f}%**")
        with col2:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)



