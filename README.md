# Traffic Sign Recognition (Deep Learning + Streamlit)

A CNN-based traffic sign classifier trained on the **GTSRB** (German Traffic
Sign Recognition Benchmark) dataset, served through an interactive
**Streamlit** web app.

## ⚠️ About the data you uploaded

Only `Train.csv`, `Test.csv`, and `Meta.csv` were provided — these are just
index files that list image **paths and labels** (e.g.
`Train/20/00020_00000_00000.png`). The actual `.png` image files (tens of
thousands of them, several hundred MB) were **not** included.

To train the model you need the full dataset with the images. The easiest
way is the Kaggle dataset:

> **GTSRB - German Traffic Sign Recognition Benchmark**
> https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Download it and arrange the files like this inside this project's `data/`
folder (the CSVs are already copied in for you):

```
traffic_sign_recognition/
└── data/
    ├── Train.csv
    ├── Test.csv
    ├── Meta.csv
    ├── Train/      <- 43 folders (0-42), each full of training images
    ├── Test/       <- test images referenced by Test.csv
    └── Meta/       <- one reference image per class
```

## Project structure

```
traffic_sign_recognition/
├── data/                 # CSVs (+ image folders you add)
├── labels.py             # 43 class-id -> sign-name mapping
├── train_model.py        # builds, trains, evaluates and saves the CNN
├── app.py                # Streamlit inference app
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

Once `data/Train`, `data/Test`, and `data/Meta` contain the actual images:

```bash
python train_model.py
```

This will:
- Load and resize every training/test image to 32x32 RGB
- Train a CNN (3 conv blocks + batch norm + dropout) with data augmentation
- Evaluate on the held-out `Test.csv` split
- Save the trained model as `traffic_sign_model.h5`

Training on the full ~39,000-image dataset typically reaches **95%+ test
accuracy** within 15-20 epochs. A GPU is recommended but not required.

## 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a photo of a traffic sign, and the app will:
- Show the predicted sign name and class ID
- Show the model's confidence
- Display the top-5 most likely classes as a chart
- Show the official reference sign image from `Meta.csv`

## Model architecture

- 3 convolutional blocks (32 → 64 → 128 filters), each with
  Conv2D + BatchNorm + MaxPooling + Dropout
- Fully connected layer (256 units) with BatchNorm + Dropout
- Softmax output over 43 classes
- Trained with Adam optimizer, categorical cross-entropy loss,
  `ImageDataGenerator` augmentation (rotation, zoom, shift, shear),
  early stopping and learning-rate reduction on plateau

## Notes

- `IMG_SIZE` (32x32) and the CNN depth in `train_model.py` can be increased
  for higher accuracy at the cost of training time.
- If you don't have a GPU, reduce `EPOCHS` in `train_model.py` for a quicker
  first run, or train on a subset of classes to validate the pipeline.
