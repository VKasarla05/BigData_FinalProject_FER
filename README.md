# Facial Emotion Classification using CNN & Spark (FER2013)

This project implements an end-to-end Spark-assisted preprocessing pipeline and CNN-based deep-learning models to classify facial emotions using the FER2013 dataset. The work compares a Baseline CNN, MobileNetV2 Transfer Learning, and Fine-Tuned MobileNetV2, and also includes Grad-CAM visual explanations.

# Dataset

FER2013 (35,887 grayscale 48×48 images, 7 emotion classes)
Kaggle Link: https://www.kaggle.com/datasets/msambare/fer2013

# Pipeline Overview

Spark-based distributed preprocessing

Image resizing, normalization, and stratified train/val/test split

Baseline CNN from scratch

MobileNetV2 transfer learning + fine-tuning

Grad-CAM & Grad-CAM++ interpretability

Model comparison and evaluation

# Results (Test Accuracy)

Baseline CNN: 55%

MobileNetV2: 48%

Fine-Tuned MobileNetV2: 47%
