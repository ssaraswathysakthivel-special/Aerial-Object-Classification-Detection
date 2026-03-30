# Aerial-Object-Classification-Detection
Project Overview
This project classifies aerial images captured by drones or satellites. The goal is to identify objects such as drones, birds, or other aerial objects using deep learning.

It combines:
Custom CNN for baseline classification
Transfer Learning (EfficientNetB0) with fine-tuning for better accuracy

Features:
Custom CNN for lightweight baseline classification.
Transfer Learning using EfficientNetB0 pretrained on ImageNet.
Fine-tuning of pretrained layers to adapt to aerial images.
Data augmentation to improve model robustness.
Tracks accuracy, precision, recall during training.
Saves best model checkpoints for deployment.
Compatible with Streamlit for live demo.

Dataset
Aerial images dataset containing multiple classes (e.g., drone, bird).
Split into training, validation, and test sets:
Training: used to train model
Validation: used for early stopping and hyperparameter tuning
Test: used only for final evaluation

Data Augmentation Techniques:
Horizontal/vertical flips
Random rotation
Zoom in/out
Brightness adjustment

Requirements
Python 3.9+
TensorFlow 2.20.0
NumPy, Pandas
Matplotlib, Seaborn (optional, for visualization)
Streamlit (for deployment)


Model Training Workflow

1. Transfer Learning
Load EfficientNetB0 (ImageNet weights)
Freeze base model
Train top layers on your dataset

2. Fine-Tuning
Unfreeze last 20 layers
Recompile with small learning rate (1e-5)
Train again with training + validation sets

3. Evaluation
Evaluate on test set only
Metrics: Accuracy, Precision, Recall, Confusion Matrix

Deployment
Save fine-tuned model:


Challenges & Solutions
Small dataset / class imbalance	- need Data augmentation
Drones confused with birds	- need Fine-tuned EfficientNet last layers
Tiny objects in aerial images	- Use higher resolution or object cropping






Metrics: Accuracy, Precision, Recall, Confusion Matrix
Deployment
Save fine-tuned model:
