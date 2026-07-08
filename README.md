# Plant Disease Detection System

An AI-powered web application that detects plant diseases from leaf images using deep learning. The application predicts the disease, displays the confidence score, provides treatment recommendations, and allows users to compare their uploaded image with healthy/diseased sample images.

---

## Features

-Upload a leaf image for disease detection
-Disease prediction using MobileNetV2
-Displays prediction confidence
-Treatment recommendations for detected diseases
-Compare uploaded image with example images
-Simple and interactive Streamlit interface

---

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow

---

## Project Structure

```
Plant-Disease-Detection-System/
│
├── app.py
├── treatment.py
├── class_names.json
├── requirements.txt
├── models/
│   ├── mobilenetv2_model.keras
│   └── multicrop_mobilenetv2.keras
├── examples/
├── README.md
└── .gitignore

---

## Installation

Clone the repository

```bash
git clone https://github.com/rehmatchawla22/Plant-Disease-Detection-System.git
```

Move into the project folder

```bash
cd Plant-Disease-Detection-System


Install dependencies

```bash
pip install -r requirements.txt

Run the application

```bash
streamlit run app.py
```

---

## Model

The project uses a MobileNetV2-based deep learning model trained on the PlantVillage dataset for plant disease classification.

---

## Future Improvements

- Grad-CAM Heatmaps
- Multi-language support
- Real-time camera detection
- Cloud deployment
- Disease history tracking

