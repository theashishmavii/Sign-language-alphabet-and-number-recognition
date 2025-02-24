# ------------------------------- Import Necessary Libraries -------------------------------

from keras.models import load_model  # Load trained Keras model
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from collections import deque  # Deque for maintaining a buffer of predictions
import time  # Time module for handling time-based updates

# ------------------------------- Load the Pretrained Model -------------------------------

model = load_model("C:/Sign-language Project/cnn_alphabet_and_number_model.h5")  # Load trained model
image_size = 64  # Define image size for resizing

# ------------------------------- Define Labels Mapping -------------------------------

labels_dict = {i: chr(65 + i) for i in range(26)}  # Mapping A-Z (0-25)
labels_dict.update({26 + i: str(i + 1) for i in range(9)})  # Mapping 1-9 (26-34)

# ------------------------------- Initialize Text Buffer -------------------------------

text_buffer = deque(maxlen=20)  # Buffer to store recent predictions
typed_text = ""  # Store the sentence being formed
last_update_time = time.time()  # Timer for updating text output

# ------------------------------- Capture Video from Webcam -------------------------------

cap = cv2.VideoCapture(0)  # Initialize webcam for real-time video capture

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        break  # Exit loop if no frame is captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # ------------------------------- Define Region of Interest (ROI) -------------------------------

    roi_size = 250  # Define size of the ROI
    x = frame.shape[1] - roi_size - 50  # X-coordinate of ROI (right side)
    y = 80  # Y-coordinate of ROI
    w, h = roi_size, roi_size  # Width and height of ROI
    roi = gray[y:y+h, x:x+w]  # Extract region of interest

    # ------------------------------- Preprocess the ROI -------------------------------

    roi_resized = cv2.resize(roi, (image_size, image_size))  # Resize ROI to model input size
    roi_resized = roi_resized.astype("float32") / 255.0  # Normalize pixel values
    roi_resized = np.expand_dims(roi_resized, axis=-1)  # Add channel dimension
    roi_resized = np.expand_dims(roi_resized, axis=0)  # Expand batch dimension

    # ------------------------------- Make Prediction -------------------------------

    prediction = model.predict(roi_resized)  # Get model prediction
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = np.max(prediction)  # Get confidence score

    # Determine prediction label
    if confidence > 0.6:
        predicted_label = labels_dict.get(predicted_class, "Unknown")  # Map class to label
    else:
        predicted_label = "Uncertain"  # Assign "Uncertain" if confidence is low

    text_buffer.append(predicted_label)  # Store predicted label in buffer
    display_text = max(set(text_buffer), key=text_buffer.count)  # Get most frequent prediction

    # ------------------------------- Update Typed Sentence -------------------------------

    if display_text != "Uncertain" and time.time() - last_update_time > 2:
        typed_text += display_text  # Append to sentence
        last_update_time = time.time()  # Reset update timer

    # ------------------------------- Display ROI and Predictions -------------------------------

    frame[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Overlay grayscale ROI on frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around ROI

    # Display prediction label and confidence score
    cv2.putText(frame, f"Prediction: {display_text} ({confidence:.2f})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display typed text
    cv2.putText(frame, f"Typed: {typed_text}", (50, frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Sign Language Detection", frame)  # Show video frame with predictions

    # ------------------------------- Exit Condition -------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop when 'q' is pressed

# ------------------------------- Release Resources -------------------------------

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close OpenCV windows
