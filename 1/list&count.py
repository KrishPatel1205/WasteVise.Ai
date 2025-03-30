import torch
import cv2
import numpy as np
import requests
from collections import Counter
from ultralytics import YOLO

model_path = "./my_model.pt"
model = YOLO(model_path)

def load_image(image_path): 
    if image_path.startswith("http"):
        resp = requests.get(image_path, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path)
    return image

def detect_food_ingredients(image_path):
    image = load_image(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return {}

    results = model(image)
    object_counts = Counter()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id < len(model.names):
                label = model.names[class_id]
                object_counts[label] += 1

    return dict(object_counts)

img = "https://m.media-amazon.com/images/I/4109t2-iaPL.jpg"
food_items = detect_food_ingredients(img)
print("Identified Food Ingredients with Counts:", food_items)
