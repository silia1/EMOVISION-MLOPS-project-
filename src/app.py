# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ─── 0. CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/best_model.keras"
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTIONS      = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE      = (48, 48)

# ─── 1. LOAD MODEL & CASCADE ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return model, face_cascade

model, face_cascade = load_resources()

st.set_page_config(
    page_title="Emotion Recognition",
    layout="wide",
)

st.title("🎭 Emotion Recognition Demo")
st.markdown(
    """
    **Sprint 5** – UI pour charger une image ou utiliser la webcam et prédire l'émotion.
    """
)

mode = st.sidebar.selectbox("Mode", ["Upload Image", "Webcam"])

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
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        face_in = preprocess_face(roi)
        preds = model.predict(face_in, verbose=0)[0]
        idx  = np.argmax(preds)
        label = EMOTIONS[idx]
        conf  = preds[idx]
        # draw
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        text = f"{label} ({conf*100:.0f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return frame, faces

# ─── 2. UPLOAD MODE ─────────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Choisissez une image…", type=["png","jpg","jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))
        annotated, faces = predict_and_annotate(image.copy())
        col1, col2 = st.columns(2)
        col1.image(image, caption="Image originale", use_column_width=True)
        if len(faces)>0:
            col2.image(annotated, caption="Avec prédiction", use_column_width=True)
        else:
            col2.warning("Aucun visage détecté")

# ─── 3. WEBCAM MODE ─────────────────────────────────────────────────────────────
else:
    st.info("Appuyez sur **Enable camera** puis attendez la capture.")
    cam_image = st.camera_input("📷 Webcam")
    if cam_image:
        image = np.array(Image.open(cam_image).convert("RGB"))
        annotated, faces = predict_and_annotate(image.copy())
        col1, col2 = st.columns(2)
        col1.image(image, caption="Webcam original", use_column_width=True)
        if len(faces)>0:
            col2.image(annotated, caption="Avec prédiction", use_column_width=True)
        else:
            col2.warning("Aucun visage détecté")

st.markdown("---")
st.caption("© Sprint 5 – Interface pour reconnaissance d’émotions")
