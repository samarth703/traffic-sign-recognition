"""
Train a CNN on the GTSRB dataset (Train.csv / Test.csv / Meta.csv + image folders)
and save the trained model as traffic_sign_model.h5 for the Streamlit app.

Expected folder layout (place the full GTSRB dataset here, e.g. from Kaggle:
"GTSRB - German Traffic Sign Recognition Benchmark"):

    data/
        Train.csv
        Test.csv
        Meta.csv
        Train/            <- 43 sub-folders of training images (0 ... 42)
        Test/              <- test images referenced by Test.csv
        Meta/              <- one representative image per class

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

from labels import NUM_CLASSES, IMG_SIZE

DATA_DIR = "data"
MODEL_PATH = "traffic_sign_model.h5"
EPOCHS = 20
BATCH_SIZE = 64


def load_split(csv_path, data_dir):
    """Load images + labels for a Train.csv / Test.csv style file."""
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row["Path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(row["ClassId"])
        except (FileNotFoundError, OSError):
            # Skip any row whose image file is missing
            continue

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading training data...")
    X, y = load_split(os.path.join(DATA_DIR, "Train.csv"), DATA_DIR)
    print(f"Loaded {len(X)} training images.")

    print("Loading test data...")
    X_test, y_test = load_split(os.path.join(DATA_DIR, "Test.csv"), DATA_DIR)
    print(f"Loaded {len(X_test)} test images.")

    y_cat = to_categorical(y, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
    )
    datagen.fit(X_train)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
