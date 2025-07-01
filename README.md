# Skin Cancer Detection using CNN

## Project Overview

This project implements a deep learning-based skin cancer detection system using **Convolutional Neural Networks (CNNs)**. The model is trained on dermoscopic images to identify and classify the type of skin cancer. The primary objective is to support early detection and diagnosis of skin cancer by automating the classification process based on image input.

## Features

- Detects and classifies the type of skin cancer from dermoscopic images.
- Built using CNN architecture for efficient image processing.
- Trained on a labeled skin cancer dataset.
- Designed for research, educational, and medical assistance purposes.

## Skin Cancer Types Detected

The model can classify images into the following skin cancer categories (based on the dataset used):

- Melanoma
- Basal Cell Carcinoma (BCC)
- Actinic Keratoses
- Benign Keratosis
- Dermatofibroma
- Vascular Lesions
- Melanocytic Nevi

> *Note: The exact number and names of classes can vary depending on the dataset.*

## Dataset

We use the **HAM10000** ("Human Against Machine with 10000 training images") dataset, which contains dermoscopic images of common pigmented skin lesions.

**Source**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV (for image preprocessing)
- Google Colab / Jupyter Notebook

## How It Works

1. Load and preprocess image data.
2. Train a CNN model on the labeled dataset.
3. Evaluate the model using test data.
4. Predict the type of skin cancer from new input images.

## Model Architecture (Example)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Change 7 to match your number of classes
])
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/skin-cancer-detection-cnn.git
cd skin-cancer-detection-cnn
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```python
python train_model.py
```

4. Predict using test image:

```python
python predict.py --image path_to_image.jpg
```

## Folder Structure

```
skin-cancer-detection/
│
├── dataset/
│   ├── train/
│   └── test/
├── models/
├── train_model.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

## Future Enhancements

- Integrate web interface for easier image upload and prediction.
- Use transfer learning for better accuracy.
- Deploy model with Flask or FastAPI.
- Add Explainable AI (XAI) visualizations like Grad-CAM.

## Disclaimer

This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis or treatment.

---
