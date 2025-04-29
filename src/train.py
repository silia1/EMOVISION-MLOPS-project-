#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ─── 1. CONFIG ────────────────────────────────────────────────────────────────
TRAIN_NPZ  = "processed_data2/train.npz"
TEST_NPZ   = "processed_data2/test.npz"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE    = 64
EPOCHS        = 50
VAL_RATIO     = 0.1
INPUT_SHAPE   = (48, 48, 1)
NUM_CLASSES   = 7  # MODIFIÉ à 5
LEARNING_RATE = 1e-4
SEED          = 42

# ─── 2. DATA ───────────────────────────────────────────────────────────────────
def load_data(path):
    data = np.load(path)
    X = data["images"][..., np.newaxis]
    y = data["labels"]
    return X, y

X, y = load_data(TRAIN_NPZ)
X_test, y_test = load_data(TEST_NPZ)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_RATIO, stratify=y, random_state=SEED, shuffle=True
)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(class_weights))

# ─── 3. MODÈLE CNN ─────────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE, padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Dropout(0.25),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─── 4. CALLBACKS ──────────────────────────────────────────────────────────────
cb_early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
cb_ckpt = callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "model10_best.keras"),
    monitor="val_accuracy", save_best_only=True
)

# ─── 5. ENTRAINEMENT ───────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[cb_early, cb_ckpt],
    verbose=2
)

# ─── 6. ÉVALUATION ─────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ─── 7. MATRICE DE CONFUSION ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Matrice de Confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.colorbar()
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_model10.png"))
print(f"📊 Matrice sauvegardée dans {OUTPUT_DIR}/confusion_matrix_model10.png")
