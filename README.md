# 🖐 Sign Language to Text Conversion using CNN

## 📌 Overview

This project implements a **Sign Language to Text Conversion System** using **Convolutional Neural Networks (CNNs)**. It includes:

- **Alphabet & Number Recognition Model** (`cnn_alphabet_and_number_model.h5`)   

This model allow real-time and image-based predictions for sign language.

---

## 📌 Features

✔ **Alphabet & Number Recognition** (A-Z, 1-9)  
✔ **Word-Based Sign Recognition** (131 ISL words)  
✔ **Real-time Sign Detection via Webcam**  
✔ **Batch Image Prediction for Testing**  
✔ **Deep Learning Model using CNN**

---

## 📂 Folder Structure  

📦 **Sign-language-alphabet-and-number-recognition**  
├── 📂 **dataset/**              # Training dataset  
├── 📂 **model/**                 # Trained AI models  
├── 📂 **scripts/**                # Preprocessing, training & alert scripts  
│   ├── 📜 **Train_model_on_alphabet.py**        # Training script  
│   ├── 📜 **image_sign2text.py**         # Model testing and evaluation  
│   ├── 📜 **live_alphabet_sign2text.py**  # Real-time sign to text conversion using model  
├── 📜 **requirements.txt**        # Python dependencies  
└──📜 **README.md**               # Project documentation  
  

