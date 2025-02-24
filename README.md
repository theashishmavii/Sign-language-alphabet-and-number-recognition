# 🖐 Sign Language to Text Conversion using CNN

## 📌 Overview

This project implements a **Sign Language to Text Conversion System** using **Convolutional Neural Networks (CNNs)**. It includes:

- **Alphabet & Number Recognition Model** (`cnn_alphabet_and_number_model.h5`)  

This model allows **real-time and image-based predictions** for sign language, enabling **seamless communication** for individuals with hearing and speech impairments.

---

## 📌 Why This Project?

Millions of people worldwide rely on **sign language** for communication, yet most of society is unfamiliar with it. This creates a **communication gap** between individuals who use sign language and those who do not.  
The **goal** of this project is to bridge this gap by providing a **real-time sign-to-text conversion tool** that enables:  

✔ **Better accessibility** for individuals with hearing and speech impairments.  
✔ **Increased awareness** of sign language in society.  
✔ **Potential for future integrations** in assistive technology, smart devices, and education.  

---

## 📌 Features

✔ **Alphabet & Number Recognition** (A-Z, 1-9)  
✔ **Real-time Sign Detection via Webcam**  
✔ **Batch Image Prediction for Testing**  
✔ **Deep Learning Model using CNN**  
✔ **High-Speed Processing for Instant Feedback**  
✔ **Scalability to Train on More Signs**  
✔ **Easy-to-Use Training Scripts for Custom Datasets**  

---

## 📌 Societal Benefits  

✅ **Empowering the Deaf and Mute Community** – Helps individuals communicate effectively in real-world scenarios.  
✅ **Improving Accessibility** – Can be integrated into **schools, hospitals, and workplaces** for better inclusion.  
✅ **Advancing Assistive Technology** – Can serve as a foundation for **smart AI translators and AR-based sign recognition systems**.  
✅ **Encouraging Research & Development** – Provides an **open-source framework** for developers to improve and expand.  

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

---

## 📸 Some Alpahbetic signs:
![23](https://github.com/user-attachments/assets/fffa40bc-2ad6-4fe0-a127-1881705c680d)
![24](https://github.com/user-attachments/assets/afc2cb53-bdba-4b98-9cc7-983f83eb62e2)
![23](https://github.com/user-attachments/assets/1057a35b-576e-4e63-b73c-137fee7b608f)
![23](https://github.com/user-attachments/assets/baa50593-b18d-4d93-abca-7d215fb4caa5)
![11](https://github.com/user-attachments/assets/d7d1650c-184c-439c-847e-114b5543d236)
![11](https://github.com/user-attachments/assets/45f44444-9e45-4550-9f5f-e7804a93fc1c)


---

## 📌 Technologies Used

| Component               | Technology Used        | Description |
|-------------------------|----------------------|-------------|
| **Programming Language** | Python | Used for model training, testing, and implementation |
| **Deep Learning Framework** | TensorFlow & Keras | Used to build and train the CNN model |
| **Computer Vision** | OpenCV | Used for real-time video processing and image handling |
| **Data Processing** | Pandas & NumPy | Used for dataset handling and preprocessing |
| **Machine Learning** | Scikit-learn | Used for data splitting and evaluation |
| **Visualization** | Matplotlib | Used for data visualization and accuracy plotting |

---
