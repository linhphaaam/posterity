import cv2
import numpy as np
import math
import random
import os
import pandas as pd
from CLIP_helper import *
import time

# Load concepts from the CSV file
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
file_path = THIS_FOLDER + '/unique_items.csv'
data = pd.read_csv(file_path)

# Extract the "Unique Items" column into a list and remove any NaN values
object_list = data['Unique Items'].dropna().tolist()

# Define number of objects to detect
number_of_objects = 10

# Define threshold to display detected objects based on similarity score
similarity_threshold = 0.5

# The number of concept axes
axes_num = len(object_list)

# Encode the text concepts to get the vectors
axes_vec = np.array([encode_text(axis) for axis in object_list])

# Record the rolling average of the similarities to smooth the visualization
smooth_sims = np.zeros(axes_num)

# Record the rolling maximum similarity ever to normalize the similarities
rolling_max_sim_ever = 0.0

# Initialize the camera using OpenCV
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define detection time interval variables
interval = 2  # Detection interval in seconds
last_detection_time = time.time()

# Variable to store the last detected labels
last_labels = []

# List to store all detected labels over time
all_detected_labels = []

# Define the path to save the output CSV file
output_csv_path = os.path.join(THIS_FOLDER, "detected_labels.csv")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Dynamically calculate text size and spacing
    base_font_size = 1  # Base font size
    base_thickness = 2  # Base thickness
    base_spacing = 40   # Base vertical spacing

    font_size = max(0.5, base_font_size * (10 / number_of_objects))  # Scale down font size
    font_thickness = max(1, int(base_thickness * (10 / number_of_objects)))  # Scale down thickness
    font_spacing = max(20, int(base_spacing * (10 / number_of_objects)))  # Scale down spacing

    # Check if 2 seconds have passed since the last detection
    current_time = time.time()
    if current_time - last_detection_time >= interval:
        last_detection_time = current_time  # Reset the timer

        # Encode the image to get the activation vector
        image_vec = encode_image(frame)

        # Calculate the similarity of the image to each of the concept axes
        similarities = np.dot(axes_vec, image_vec)

        # Smooth the similarities by taking a rolling average
        smooth_sims = 0.9 * smooth_sims + 0.1 * similarities

        # Normalize the similarities to be between 0 and 1
        min_sim = np.min(smooth_sims)
        max_sim = np.max(smooth_sims)
        rolling_max_sim_ever = rolling_max_sim_ever * 0.95 + max_sim * 0.05
        similarities_norm = np.abs((smooth_sims - min_sim) / (rolling_max_sim_ever - min_sim))

        # Get the indices of the top n similarities in descending order
        top_indices = np.argsort(similarities_norm)[-number_of_objects:][::-1]

        # Update the last detected labels
        last_labels = [
            f"{object_list[index]}: {similarities_norm[index]:.2f}"
            for index in top_indices if similarities_norm[index] > similarity_threshold
        ]

        # Append the detected labels to the list of all labels
        if last_labels:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
            for label in last_labels:
                all_detected_labels.append([timestamp, label])  # Add timestamp and label

        # DEBUG: Print detected labels
        print(f"Detected labels: {last_labels}")

    # Display the top detected concepts with their similarity scores
    y_offset = 50
    for label in last_labels:
        text_color = (255, 0, 100)
        label_position = (50, y_offset)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)
        y_offset += font_spacing

    # Show the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save all detected labels to a CSV file when the program ends
detected_labels_df = pd.DataFrame(all_detected_labels, columns=["Timestamp", "Detected Label"])
detected_labels_df.to_csv(output_csv_path, index=False)

print(f"Detected labels saved to {output_csv_path}")

cap.release()
cv2.destroyAllWindows()
