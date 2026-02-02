# Pneumonia Detection from Chest X-Rays using Deep Learning

## Overview
This project implements a deep learning pipeline for detecting pneumonia from chest X-ray images using a convolutional neural network (ResNet-50) and the NIH ChestXray14 dataset. The project includes model training, evaluation, and explainability using Grad-CAM.

---

## Dataset
- Source: NIH ChestXray14
- Images: Frontal chest X-rays
- Labels: Weakly labeled from radiology reports
- Task: Binary classification (Pneumonia vs Normal)

---

##  Model
- Architecture: ResNet-50 (pretrained on ImageNet)
- Loss: BCEWithLogitsLoss with class weighting
- Optimizer: Adam
- Input size: 224×224

---

## Evaluation Metrics
- ROC-AUC
- Sensitivity (Recall)
- Specificity
- Confusion Matrix

---

## Explainability
Grad-CAM is used to visualize regions of the image influencing model predictions for:
- True Positive
- False Positive
- True Negative
- False Negative

This helps assess whether the model focuses on clinically relevant lung regions.

---

##  Results

At threshold 0.5:

TP = 54  
FN = 13  
TN = 1429  
FP = 3504  

AUC ≈ 0.59  
Sensitivity ≈ 0.81  
Specificity ≈ 0.29  

The model shows high sensitivity but poor specificity, indicating limited clinical reliability.

---

## Discussion

The model exhibits weak class separation, likely due to:
- Label noise in the dataset
- Severe class imbalance
- Limited training time
- Simplified binary task

Grad-CAM visualizations suggest inconsistent attention to pathological regions.

This project demonstrates:
- Transfer learning for medical imaging
- Binary classification from X-rays
- Model evaluation and explainability

---

##  Project Structure




---

## Future Improvements
- Multi-label classification
- Better probability calibration
- Larger training set
- Use DenseNet architectures
- Improved preprocessing
- Clinical validation

---

## Disclaimer
This project is for educational purposes only and must not be used for real medical diagnosis.
