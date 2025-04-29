#!/usr/bin/env python3
import os
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from huggingface_hub import HfApi

# ─── CONFIGURATION ─────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
REPO_ID    = "azzmannorelimane/EMOVISION_DB"
OUTPUT_DIR = "processed_data2"
IMG_SIZE   = (48, 48)

# ─── FONCTIONS DE PRÉTRAITEMENT ─────────────────────────────────────────
def preprocess_split(split_name: str) -> str:
    """
    Charge le split HuggingFace, applique les transformations et enregistre
    les données traitées au format .npz.
    """
    ds = load_dataset(
        "imagefolder",
        data_dir="EMOVISION_DB",  # chemin relatif vers le clone local
        split=split_name
    )
    images, labels = [], []

    # Définition de l'augmentation des données
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        validation_split=0.2
    )

    for ex in tqdm(ds, desc=f"Prétraitement {split_name}"):
        img = ex["image"].convert("L").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(ex["label"])

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.npz")
    np.savez_compressed(out_path, images=images, labels=labels)
    print(f"✅ {split_name} → {out_path} (shape {images.shape})")
    return out_path

def upload_processed(file_path: str):
    """
    Téléverse un fichier local dans le dossier 'processed/' du repo dataset.
    """
    api = HfApi()
    filename = os.path.basename(file_path)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data2/{filename}",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"🚀 Téléversé {filename} dans {REPO_ID}/processed")

# ─── SCRIPT PRINCIPAL ───────────────────────────────────────────────────
if __name__ == "__main__":
    train_npz = preprocess_split("train")
    test_npz  = preprocess_split("test")
    upload_processed(train_npz)
    upload_processed(test_npz)
    print("🏁 Pipeline terminée !")
