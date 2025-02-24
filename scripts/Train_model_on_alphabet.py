# ------------------------------- Import Necessary Libraries -------------------------------

import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for handling datasets
import keras  # Keras for deep learning model creation
from keras import layers  # Import necessary layers for the model
import os  # OS module for file handling
import cv2  # OpenCV for image processing
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing
from keras.utils import to_categorical  # Converting labels to one-hot encoding

# ------------------------------- Define Dataset Path and Parameters -------------------------------

dataset_path = "path_to_the_xlsx_sheet or datset"  # Path to dataset
image_size = 64  # Image size for resizing
num_classes = 35  # Total number of classes (A-Z, 1-9)

# ------------------------------- Function to Load Data from Excel -------------------------------

def load_data_from_excel(excel_file):
    df = pd.read_excel(excel_file)  # Read dataset from Excel file
    images, labels = [], []  # Lists to store images and labels

    for index, row in df.iterrows():
        img_path = row['Frame Path']  # Get image file path
        label = row['Alphabet']  # Get corresponding label

        # Read and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale mode
        img = cv2.resize(img, (image_size, image_size))  # Resize image to required dimensions
        images.append(img)  # Store preprocessed image

        # Convert label to index
        if label.isalpha():  # If the label is an alphabet (A-Z)
            labels.append(ord(label.upper()) - ord('A'))  # Convert A-Z to indices (0-25)
        elif label.isdigit():  # If the label is a number (1-9)
            labels.append(26 + int(label) - 1)  # Convert 1-9 to indices (26-34)

    # Convert lists to NumPy arrays and normalize images
    images = np.array(images).reshape(-1, image_size, image_size, 1) / 255.0  # Normalize pixel values
    labels = to_categorical(np.array(labels), num_classes)  # Convert labels to one-hot encoding
    return train_test_split(images, labels, test_size=0.2, random_state=42)  # Split dataset into training and testing

# ------------------------------- Load Dataset -------------------------------

x_train, x_test, y_train, y_test = load_data_from_excel(dataset_path)  # Load dataset

# ------------------------------- Define CNN Model -------------------------------

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),  # First convolutional layer
    layers.MaxPooling2D((2, 2)),  # First pooling layer

    layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D((2, 2)),  # Second pooling layer

    layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    layers.MaxPooling2D((2, 2)),  # Third pooling layer

    layers.Flatten(),  # Flatten layer to convert 2D feature maps to 1D vector
    layers.Dense(128, activation='relu'),  # Fully connected dense layer
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# ------------------------------- Compile the Model -------------------------------

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model with Adam optimizer

# ------------------------------- Train the Model -------------------------------

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))  # Train model with dataset

# ------------------------------- Evaluate the Model -------------------------------

test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluate model performance
print(f"Test Accuracy: {test_acc:.2f}")  # Print test accuracy

# ------------------------------- Save the Trained Model -------------------------------

model.save("cnn_alphabet_and_number_model.h5")  # Save trained model for future use
