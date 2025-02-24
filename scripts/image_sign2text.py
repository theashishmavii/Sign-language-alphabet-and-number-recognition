# ------------------------------- Import Necessary Libraries -------------------------------

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical computations
from keras.models import load_model  # Load trained deep learning model

# ------------------------------- Load the Pretrained Model -------------------------------

model = load_model("C:/Sign-language Project/cnn_alphabet_and_number_model.h5")  # Load trained model for alphabet & number recognition
image_size = 64  # Define input image size expected by the model

# ------------------------------- Define Label Mappings -------------------------------

# Mapping indices to alphabets (A-Z) and numbers (1-9)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z (0-25)
labels_dict.update({26 + i: str(i + 1) for i in range(9)})  # 1-9 (26-34)

# ------------------------------- Load and Validate Image -------------------------------

# Define image path (update this according to the dataset)
image_path = "C:/Sign-language Project/datasets/ISL dataset/B/2.jpg"  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path!")
    exit()

# ------------------------------- Preprocess Image for Model -------------------------------

image_resized = cv2.resize(image, (image_size, image_size))  # Resize image to match model input
image_resized = image_resized.astype("float32") / 255.0  # Normalize pixel values to [0,1]
image_resized = np.expand_dims(image_resized, axis=-1)  # Add channel dimension for CNN
image_resized = np.expand_dims(image_resized, axis=0)   # Add batch dimension

# ------------------------------- Make Prediction -------------------------------

prediction = model.predict(image_resized)  # Get model predictions
predicted_class = np.argmax(prediction)  # Get class index with highest probability
confidence = np.max(prediction)  # Extract confidence score

# ------------------------------- Determine Final Prediction -------------------------------

# Assign predicted label only if confidence is above threshold, else mark as "Uncertain"
predicted_label = labels_dict.get(predicted_class, "Unknown") if confidence > 0.4 else "Uncertain"
prediction_text = f"Prediction: {predicted_label} ({confidence:.2f})"

# ------------------------------- Convert Grayscale to Color for Display -------------------------------

display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to color for display

# ------------------------------- Resize Image for Display -------------------------------

screen_width = 800  # Define screen width limit
max_width = min(screen_width, display_image.shape[1])  # Adjust width to fit screen
scale_factor = max_width / display_image.shape[1]  # Compute scaling factor
new_height = int(display_image.shape[0] * scale_factor)  # Compute new height
display_image = cv2.resize(display_image, (max_width, new_height))  # Resize image

# ------------------------------- Create a Canvas for Display -------------------------------

# Calculate text width dynamically for proper alignment
(text_width, text_height), _ = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
canvas_width = max(max_width, text_width + 40)  # Ensure enough space for text
canvas_height = new_height + 60  # Extra space for text below image

# Create a white background canvas
canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255  
canvas[:new_height, :max_width] = display_image  # Place resized image on the canvas

# ------------------------------- Display Prediction on Canvas -------------------------------

# Center the prediction text horizontally
text_x = (canvas_width - text_width) // 2  
cv2.putText(canvas, prediction_text, (text_x, new_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# ------------------------------- Show Image with Prediction -------------------------------

while True:
    cv2.imshow("Image Prediction", canvas)  # Display image with prediction

    # Exit loop when Enter key is pressed
    if cv2.waitKey(1) == 13:  # 13 is ASCII for Enter key
        break

# ------------------------------- Release Resources -------------------------------

cv2.destroyAllWindows()  # Close all OpenCV windows