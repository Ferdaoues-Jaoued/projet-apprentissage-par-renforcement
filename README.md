
Breast Cancer Detection using VGG16 Transfer Learning
# 🩺 Breast Cancer Detection using VGG16 Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ferdaoues-Jaoued/projet-apprentissage-par-renforcement/blob/main/notebook/breast_cancer_classification.ipynb)


########### 🚀 How to Run This Notebook

### 🌐 Using Google Colab

1. Click on the **Open in Colab** badge at the top of this page.
2. In Colab, go to **Runtime → Run All**.

Everything runs automatically.

---

## 📌 About

This repository implements a Deep Learning pipeline for **breast cancer detection** from mammography images by classifying them into:

- **Normal**
- **Benign**
- **Malignant**

The model is built using **Transfer Learning with VGG16** (pretrained on ImageNet) and enhanced with image preprocessing and data augmentation.

---


## 🧠 Pipeline Overview

### 1️⃣ Metadata Parsing

The notebook reads `Info.txt` and extracts:

- Image ID (REFNUM)
- Background type
- Lesion class (NORM, BENIGN, MALIGNANT)
- Severity
- ROI coordinates (X, Y, radius)

These are stored in a pandas DataFrame. Labels are mapped as:

| Severity | Label |
|----------|-------|
| Normal   | 0     |
| Benign   | 1     |
| Malignant| 2     |

The parsed data is saved into `info_labeled.csv`.

---

### 2️⃣ Patch Extraction

Each image is loaded with OpenCV.

- If the image has an ROI (Benign/Malignant), a patch around the ROI is cropped.
- If the image is Normal, the full image is used.
- Each patch is resized to **64×64 pixels**.
- Patches are saved in `dataset/patches/`.

---

### 3️⃣ Data Augmentation

To balance the dataset, augmentation is applied to the minority classes (Benign & Malignant) using:

- Rotation
- Horizontal flips
- Zoom
- Shear
- Width and height shifts

Augmented patches are saved in `dataset/patches_augmented/`.

---

### 4️⃣ Train / Validation / Test Split

The full set of patches (including augmented ones) is split into:

- **Train:** 80%  
- **Validation:** 10%  
- **Test:** 10%

The split is stratified to preserve class balance.  

This creates:
dataset/dataset_patches/
├── train/
├── val/
└── test/
---

### 5️⃣ Model: Transfer Learning with VGG16

VGG16 (pretrained on ImageNet)
↓ Flatten
↓ Dense(128, ReLU)
↓ Dropout(0.5)
↓ Dense(3, Softmax)

- Optimizer: **Adam**
- Learning Rate: **1e‑5**
- Loss: **Categorical Crossentropy**
- Metrics: **Accuracy**

The model is trained on the training set and validated on the validation set.

---

### 6️⃣ Evaluation

The final model is evaluated on the test set.  
Key results printed:
Test loss
Test accuracy

---








