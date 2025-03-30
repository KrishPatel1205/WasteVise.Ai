import torch
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import os

# Load YOLO model
model_path = "my_model/my_model.pt"
model = YOLO(model_path)

def detect_food_ingredients(frame):
    results = model(frame)
    detected_items = set()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item() * 100  # Convert confidence to percentage
            if class_id < len(model.names) and confidence > 80:
                label = model.names[class_id]
                detected_items.add(label)
    
    return detected_items

def update_food_log(food_items, filename="food_log.txt"):
    if not food_items:
        print("No high-confidence food items detected. Skipping file update.")
        return
    
    existing_items = set()
    
    # Read existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_items = set(file.read().splitlines())
    
    # Update log
    updated_items = existing_items.union(food_items)
    
    with open(filename, "w") as file:
        for item in updated_items:
            file.write(f"{item}\n")
        file.flush()
    print("Updated food_log.txt with:", food_items)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    food_items = detect_food_ingredients(frame)
    print("Identified Food Ingredients:", food_items)
    
    # Update log file
    update_food_log(food_items)
    
    # Display frame
    cv2.imshow("Food Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
