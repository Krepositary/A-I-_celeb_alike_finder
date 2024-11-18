import numpy as np
from PIL import Image

import cv2
print(cv2.__version__)

def preprocess_image(image_file):
    # Convert to OpenCV format
    image = Image.open(image_file)
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (160, 160))  # Required size for FaceNet
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def load_celebrity_data():
    # Load celebrity names from a JSON or CSV file
    with open("celebrity_data/celeb_names.json", "r") as f:
        celeb_names = json.load(f)
    return celeb_names
