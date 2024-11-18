pip install numpy==1.21.4 opencv-python-headless
pip install --upgrade numpy opencv-python-headless

import numpy as np
from PIL import Image

from PIL import Image
import numpy as np
import face_recognition
import io

# Preprocess the uploaded image to be used for face recognition
def preprocess_image(uploaded_file):
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    # Convert the image to RGB (if it isn't already)
    image = image.convert("RGB")
    # Convert to numpy array
    image_array = np.array(image)
    
    # Use face_recognition to find face encodings
    return image_array

