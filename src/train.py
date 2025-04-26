#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ─── 1. CONFIGURATION ──────────────────────────────────────────────────────────
TRAIN_NPZ   = "data/processed/train.npz"
TEST_NPZ    = "data/processed/test.npz"
OUTPUT_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE   = 64
EPOCHS       = 50
VAL_RATIO    = 0.1
INPUT_SHAPE  = (48, 48, 1)
NUM_CLASSES  = 7
LEARNING_RATE= 1e-3
SEED         = 42

# ─── 2. CHARGEMENT & SPLIT ─────────────────────────────────────────────────────
def load_data(path):
    data = np.load(path)
    X = data["images"][..., np.newaxis]  # (N,48,48,1)
    y = data["labels"]
    return X, y

# Chargement complet
X, y = load_data(TRAIN_NPZ)
X_test, y_test = load_data(TEST_NPZ)

# Split stratifié Train / Val
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_RATIO,
    stratify=y,
    random_state=SEED,
    shuffle=True
)

# Class weights
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): w for c, w in zip(classes, weights)}

print("Shapes →",
      "Train:", X_train.shape, y_train.shape,
      "Val:",   X_val.shape,   y_val.shape,
      "Test:",  X_test.shape,  y_test.shape)
print("Class weights:", class_weight)

# ─── 3. DÉFINITION DU MODÈLE AVEC AUGMENTATION ────────────────────────────────
def build_model(input_shape, num_classes):
    return models.Sequential([
        layers.Input(input_shape),

        # ---- Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # ---- Convolutions
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),

        # ---- Classique
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ], name="emotion_cnn")

model = build_model(INPUT_SHAPE, NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ─── 4. CALLBACKS ─────────────────────────────────────────────────────────────
cb_early   = callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
cb_plateau = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
cb_ckpt    = callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

# ─── 5. ENTRAÎNEMENT ──────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[cb_early, cb_plateau, cb_ckpt],
    shuffle=True,
    verbose=2
)

# ─── 6. ÉVALUATION FINALE ─────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Prédictions & Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables étiquettes")
plt.colorbar()
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(f"🌟 Matrice sauvegardée dans {OUTPUT_DIR}/confusion_matrix.png")
