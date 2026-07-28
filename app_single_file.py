"""
Traffic Sign Recognition - Deep Learning + Streamlit (single file)

Everything (class labels, CNN model, training, and the Streamlit UI) lives
in this one file.

Folder layout expected next to this file:

    app.py                 <- this file
    data/
        Train.csv
        Test.csv
        Meta.csv
        Train/              <- 43 folders (0-42) of training images
        Test/                <- test images referenced by Test.csv
        Meta/                <- one reference image per class

Run:
    streamlit run app.py

If data/Train and data/Test contain the real images, use the "Train Model"
button in the sidebar to train the CNN from right inside the app (this can
take a while without a GPU). Once trained, traffic_sign_model.h5 is saved
and reused automatically on every future run.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------------------------------
# Config & class labels (GTSRB - 43 classes)
# --------------------------------------------------------------------------

DATA_DIR = "data"
MODEL_PATH = "traffic_sign_model.h5"
IMG_SIZE = 32
EPOCHS = 20
BATCH_SIZE = 64

CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}
NUM_CLASSES = len(CLASS_NAMES)

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

# --------------------------------------------------------------------------
# Data loading & CNN (used only when training)
# --------------------------------------------------------------------------

def load_split(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row["Path"])
        try:
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(row["ClassId"])
        except (FileNotFoundError, OSError):
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
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_model(progress_callback=None):
    X, y = load_split(os.path.join(DATA_DIR, "Train.csv"), DATA_DIR)
    X_test, y_test = load_split(os.path.join(DATA_DIR, "Test.csv"), DATA_DIR)

    if len(X) == 0:
        raise RuntimeError(
            "No training images found. Make sure data/Train/ contains the "
            "actual GTSRB image files referenced by data/Train.csv."
        )

    y_cat = to_categorical(y, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES) if len(X_test) else None

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, zoom_range=0.15,
        width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
    )
    datagen.fit(X_train)

    model = build_model()

    class StreamlitProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, EPOCHS, logs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        StreamlitProgress(),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )

    if X_test is not None and len(X_test):
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    else:
        test_acc = None

    model.save(MODEL_PATH)
    return test_acc


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_meta():
    meta_csv = os.path.join(DATA_DIR, "Meta.csv")
    return pd.read_csv(meta_csv) if os.path.exists(meta_csv) else None


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def main():
    st.title("🚦 Traffic Sign Recognition")
    st.write(
        "Upload an image of a traffic sign and this CNN predicts which of "
        "the 43 GTSRB sign classes it belongs to."
    )

    with st.sidebar:
        st.header("Model")
        st.metric("Number of classes", NUM_CLASSES)

        data_ready = os.path.exists(os.path.join(DATA_DIR, "Train.csv"))
        st.write("Data folder found." if data_ready else "⚠️ `data/` folder not found.")

        if st.button("🔁 Train / Retrain model"):
            if not data_ready:
                st.error("Add data/Train.csv, data/Test.csv and the image folders first.")
            else:
                progress_bar = st.progress(0.0)
                status = st.empty()

                def on_epoch(epoch, total, logs):
                    progress_bar.progress(epoch / total)
                    status.write(
                        f"Epoch {epoch}/{total} — "
                        f"acc: {logs.get('accuracy', 0):.3f}, "
                        f"val_acc: {logs.get('val_accuracy', 0):.3f}"
                    )

                with st.spinner("Training CNN — this can take a while without a GPU..."):
                    try:
                        test_acc = train_and_save_model(progress_callback=on_epoch)
                        st.cache_resource.clear()
                        if test_acc is not None:
                            st.success(f"Training complete. Test accuracy: {test_acc:.3f}")
                        else:
                            st.success("Training complete.")
                    except RuntimeError as e:
                        st.error(str(e))

    model = load_trained_model()
    meta_df = load_meta()

    if model is None:
        st.warning(
            f"No trained model found (`{MODEL_PATH}`). Add the full GTSRB "
            "image data to `data/` and click **Train / Retrain model** in "
            "the sidebar first."
        )
        st.stop()

    uploaded_file = st.file_uploader("Choose a traffic sign image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            preds = model.predict(preprocess(image), verbose=0)[0]

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        with col2:
            st.subheader("Prediction")
            st.success(f"**{CLASS_NAMES[top_idx]}**")
            st.write(f"Class ID: `{top_idx}`")
            st.write(f"Confidence: `{confidence * 100:.2f}%`")

            if meta_df is not None:
                meta_row = meta_df[meta_df["ClassId"] == top_idx]
                if not meta_row.empty:
                    meta_path = os.path.join(DATA_DIR, meta_row.iloc[0]["Path"])
                    if os.path.exists(meta_path):
                        st.image(meta_path, caption="Reference sign", width=120)

        st.subheader("Top 5 predictions")
        top5_idx = np.argsort(preds)[-5:][::-1]
        results = pd.DataFrame({
            "Sign": [CLASS_NAMES[i] for i in top5_idx],
            "Confidence (%)": [round(float(preds[i]) * 100, 2) for i in top5_idx],
        })
        st.bar_chart(results.set_index("Sign"))
        st.dataframe(results, hide_index=True, use_container_width=True)
    else:
        st.info("Upload an image to get a prediction.")


if __name__ == "__main__":
    main()
