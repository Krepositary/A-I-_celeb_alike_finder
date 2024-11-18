import numpy as np
import cv2
print(cv2.__version__)
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
face_model = load_model("path/to/your/facenet_model.h5")

# Load celebrity embeddings
celeb_embeddings = np.load("celebrity_data/celeb_embeddings.npy")
celeb_names = load_celebrity_data()

def find_celebrity_match(image):
    # Extract features from the image
    image_embedding = face_model.predict(image)
    
    # Compute similarity with celebrity embeddings
    similarities = cosine_similarity(image_embedding, celeb_embeddings)
    best_match_idx = np.argmax(similarities)
    
    return celeb_names[best_match_idx], similarities[0][best_match_idx]

def generate_funny_caption(celebrity_name, similarity_score):
    captions = [
        f"You look just like a younger version of {celebrity_name}!",
        f"{celebrity_name} wishes they looked this good!",
        f"Are you secretly {celebrity_name} in disguise?",
    ]
    return captions[int(similarity_score * len(captions)) % len(captions)]
