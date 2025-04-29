import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Recognition", layout="wide", page_icon="🎭")
MODEL_PATH = "models/model10_best.keras"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = (48, 48)

# ─── LOAD MODEL & CASCADE ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return model, face_cascade

model, face_cascade = load_resources()

# ─── UI HEADER ─────────────────────────────────────────────────────────────────
st.title("🎭 Emotion Recognition with Deep Learning")
st.markdown("**Sprint 5** · UI avancée pour détection des émotions depuis webcam ou image.")
tabs = st.tabs(["📷 Webcam", "🖼️ Image Upload", "📊 Performances"])

# ─── UTILITAIRES ───────────────────────────────────────────────────────────────
def preprocess_face(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, IMG_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, -1)   # (48,48,1)
    face = np.expand_dims(face, 0)    # (1,48,48,1)
    return face

def predict_and_annotate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    all_data = []

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        face_in = preprocess_face(roi)
        preds = model.predict(face_in, verbose=0)[0]
        idx = np.argmax(preds)
        label = EMOTIONS[idx]
        confidence = preds[idx]

        # Annotate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.0f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        all_data.append((label, confidence, preds, (x, y, w, h)))

    return frame, faces, all_data

def plot_probabilities(preds):
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.barplot(x=EMOTIONS, y=preds, ax=ax, palette="viridis")
    ax.set_title("Distribution des probabilités")
    ax.set_ylabel("Probabilité")
    ax.set_ylim([0, 1])
    for i, v in enumerate(preds):
        ax.text(i, v + 0.01, f"{v:.2f}", color='black', ha='center')
    st.pyplot(fig)

# ─── 1. WEBCAM ─────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🎥 Capture Webcam")
    st.info("Activez votre webcam et appuyez sur 'Prendre une photo'.")

    cam_image = st.camera_input("📸 Capture via webcam")
    if cam_image:
        image = np.array(Image.open(cam_image).convert("RGB"))
        annotated, faces, results = predict_and_annotate(image.copy())

        st.image(annotated, caption="Image annotée", use_column_width=True)
        if results:
            for label, conf, preds, _ in results:
                st.success(f"**Émotion détectée : `{label}`** avec une confiance de **{conf*100:.1f}%**")
                plot_probabilities(preds)
        else:
            st.warning("Aucun visage détecté.")

# ─── 2. IMAGE UPLOAD ───────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("📁 Téléverser une image")
    uploaded = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))
        annotated, faces, results = predict_and_annotate(image.copy())

        col1, col2 = st.columns(2)
        col1.image(image, caption="Image originale", use_column_width=True)
        col2.image(annotated, caption="Image avec prédictions", use_column_width=True)

        if results:
            for label, conf, preds, _ in results:
                st.success(f"**Émotion prédite : `{label}`** ({conf*100:.1f}%)")
                plot_probabilities(preds)
        else:
            st.warning("Aucun visage détecté dans l’image.")

# ─── 3. PERFORMANCE GRAPHS ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Analyse des performances du modèle")

    history_path = "models/history_model10.npy"
    cm_path = "models/confusion_matrix_model10.png"

    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()

        # Accuracy
        st.markdown("#### Courbes d'Accuracy")
        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Train Accuracy')
        ax.plot(history['val_accuracy'], label='Val Accuracy')
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        # Loss
        st.markdown("#### Courbes de Loss")
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Val Loss')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("📉 Aucune donnée d'entraînement trouvée.")

    # Confusion matrix
    if os.path.exists(cm_path):
        st.markdown("#### Matrice de Confusion")
        st.image(cm_path, use_column_width=True)
    else:
        st.info("Pas de matrice de confusion disponible.")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 · Reconnaissance des émotions · UI améliorée avec Streamlit ✨")
