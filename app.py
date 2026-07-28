"""
Streamlit app: Traffic Sign Recognition
Upload a traffic sign image and the trained CNN predicts its class.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from labels import CLASS_NAMES, IMG_SIZE

MODEL_PATH = "traffic_sign_model.h5"
META_CSV = os.path.join("data", "Meta.csv")

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_meta():
    if os.path.exists(META_CSV):
        return pd.read_csv(META_CSV)
    return None


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🚦 Traffic Sign Recognition")
    st.write(
        "Upload an image of a traffic sign and this CNN model will "
        "predict which of the 43 GTSRB sign classes it belongs to."
    )

    model = load_model()
    meta_df = load_meta()

    if model is None:
        st.error(
            f"Trained model file `{MODEL_PATH}` not found. "
            "Run `python train_model.py` first to train and save the model."
        )
        st.stop()

    with st.sidebar:
        st.header("About")
        st.write(
            "This app uses a Convolutional Neural Network trained on the "
            "German Traffic Sign Recognition Benchmark (GTSRB) dataset "
            "(43 classes, ~39,000 training images)."
        )
        st.metric("Number of classes", len(CLASS_NAMES))

    uploaded_file = st.file_uploader(
        "Choose a traffic sign image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            batch = preprocess(image)
            preds = model.predict(batch, verbose=0)[0]

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
                    meta_path = os.path.join("data", meta_row.iloc[0]["Path"])
                    if os.path.exists(meta_path):
                        st.image(meta_path, caption="Reference sign", width=120)

        st.subheader("Top 5 predictions")
        top5_idx = np.argsort(preds)[-5:][::-1]
        results = pd.DataFrame(
            {
                "Sign": [CLASS_NAMES[i] for i in top5_idx],
                "Confidence (%)": [round(float(preds[i]) * 100, 2) for i in top5_idx],
            }
        )
        st.bar_chart(results.set_index("Sign"))
        st.dataframe(results, hide_index=True, use_container_width=True)
    else:
        st.info("Upload an image to get a prediction.")


if __name__ == "__main__":
    main()
