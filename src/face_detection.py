import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model('models/model10_best.keras')

# Labels à adapter selon votre dataset (modifiez si vous n'avez pas 7 émotions)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialiser la webcam
cap = cv2.VideoCapture(0)

# Charger le classifieur Haar pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("✅ Webcam démarrée... Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erreur de capture vidéo.")
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)  # (48, 48, 1)
        face = np.expand_dims(face, axis=0)   # (1, 48, 48, 1)

        prediction = model.predict(face, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]

        # Affichage sur le flux vidéo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.imshow('Détection des émotions en temps réel', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
