import numpy as np
from PIL import Image

import cv2
print(cv2.__version__)

from PIL import Image
import numpy as np

def preprocess_image(image_file):
    # Open the image using PIL
    image = Image.open(image_file)
    image = image.convert("RGB")
    image = image.resize((160, 160))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)


def load_celebrity_data():
    # Load celebrity names from a JSON or CSV file
    with open("celebrity_data/celeb_names.json", "r") as f:
        celeb_names = json.load(f)
    return celeb_names
