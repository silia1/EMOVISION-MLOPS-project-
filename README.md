## **Structure de Projet recommandée**

```
emotion-face-app/
│
├── data/                      # 📦 Datasets bruts et prétraités
│   ├── raw/                   #   ➤ Dataset brut (ex: fer2013.csv)
│   ├── processed/             #   ➤ Dataset nettoyé (images, numpy, etc.)
│   └── reports/               #   ➤ Histogrammes, statistiques, visualisations
│
├── notebooks/                 # 📒 Jupyter Notebooks d’exploration et essais
│   ├── 01-exploration.ipynb   #   ➤ Analyse des données
│   ├── 02-model-building.ipynb#   ➤ Essais de modèles
│   └── 03-evaluation.ipynb    #   ➤ Visualisation des performances
│
├── src/                       # ⚙️ Code source principal (modulaire)
│   ├── __init__.py
│   ├── config.py              #   ➤ Paramètres globaux (paths, config modèle)
│   ├── data_loader.py         #   ➤ Chargement, nettoyage et split du dataset
│   ├── preprocessing.py       #   ➤ Normalisation, transformation images
│   ├── model.py               #   ➤ Architecture du modèle CNN
│   ├── train.py               #   ➤ Fonction d’entraînement du modèle
│   ├── evaluate.py            #   ➤ Évaluation du modèle (matrice, metrics)
│   ├── emotion_detector.py    #   ➤ Prédiction à partir d’image ou vidéo
│   └── face_detection.py      #   ➤ OpenCV Haar cascade ou MTCNN
│
├── app/                       # 🌐 Interface utilisateur
│   ├── streamlit_app.py       #   ➤ Interface Streamlit (upload image ou webcam)
│   └── assets/                #   ➤ Images statiques, icônes, etc.
│
├── models/                    # 🧠 Modèles entraînés sauvegardés
│   ├── emotion_model.h5       #   ➤ Poids du modèle CNN
│   └── label_encoder.pkl      #   ➤ Encodeur pour les étiquettes
│
├── tests/                     # ✅ Tests unitaires
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_face_detection.py
│
├── requirements.txt           # 📌 Dépendances (pip freeze)
├── README.md                  # 📝 Description du projet
├── .gitignore                 # 🙈 Fichiers/dossiers à ignorer
└── LICENSE                    # 📄 (optionnel) Licence open source
```
