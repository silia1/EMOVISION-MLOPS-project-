import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model('models/best_model.keras')

# Labels correspondant aux émotions (à adapter selon ton dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialiser la webcam
cap = cv2.VideoCapture(0)  # 0 = première webcam

# Charger un détecteur de visage pré-entraîné (Haar Cascade de OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("✅ Webcam démarrée... Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture vidéo.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Détecter les visages

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extraire la zone du visage
        face = cv2.resize(face, (48, 48))  # Redimensionner comme dans ton dataset
        face = face.astype('float32') / 255.0  # Normaliser entre 0 et 1
        face = np.expand_dims(face, axis=-1)   # (48,48,1)
        face = np.expand_dims(face, axis=0)    # (1,48,48,1)

        prediction = model.predict(face)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]

        # Dessiner le rectangle et le label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Afficher la frame
    cv2.imshow('Détection des émotions', frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
