# 🌍 Deep Learning Approach for Land Cover Type Classification for Natural Disaster Applications

## 📌 Project Overview
This project presents an intelligent system for **Land Cover Type Classification** using **Deep Learning (CNN)** applied to **satellite imagery** for natural disaster analysis.  

The system is designed to classify land cover types affected by disasters such as:
- 🌊 Flood
- 🔥 Wildfire
- 🌪️ Cyclone
- 🌍 Earthquake  

By leveraging **Convolutional Neural Networks (CNNs)** with **Transfer Learning (ResNet18)**, the system provides accurate and efficient classification of disaster-affected regions.

A **Tkinter-based GUI** is integrated to allow users to interact with the system easily, making it suitable for **disaster management teams, researchers, and non-technical users**.

---

## 🎯 Problem Statement
Traditional land cover classification systems:
- Rely on manual feature extraction  
- Use classical ML models (SVM, KNN, Random Forest)  
- Lack accuracy for complex satellite imagery  
- Do not support real-time predictions  
- Require expert knowledge  

➡️ This project solves these limitations using **deep learning + automation + GUI-based interaction**.

---

## 💡 Proposed Solution
The proposed system:
- Uses **CNN (ResNet18)** for feature extraction and classification  
- Applies **transfer learning** for faster and accurate training  
- Provides **real-time prediction capability**  
- Includes a **user-friendly GUI** for easy operation  
- Supports **automated preprocessing and evaluation**  

---

## 🚀 Key Features
- 📂 Dataset upload through GUI  
- 🔄 Image preprocessing (resize, normalize)  
- 🧠 CNN model training using PyTorch  
- 📊 Performance evaluation:
  - Accuracy
  - Confusion Matrix
  - Classification Report  
- 🖼️ Real-time prediction with OpenCV visualization  
- 💾 Model saving (`cnn_model.pth`)  
- 🖥️ Fully interactive GUI  

---

## 🛠️ Tech Stack

### 🔹 Programming & Frameworks
- Python  
- PyTorch  
- Torchvision  

### 🔹 Libraries
- NumPy, Matplotlib, Seaborn  
- OpenCV, PIL  
- Scikit-learn  

### 🔹 GUI
- Tkinter  

---

## 📁 Dataset Structure
The dataset must follow this structure:
