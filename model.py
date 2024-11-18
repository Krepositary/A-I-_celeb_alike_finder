import face_recognition
import numpy as np

# Preload celebrity data
def load_celebrity_data():
    # Load your celebrity images and encode their faces
    celebrities = {
        "celebrity_name_1": face_recognition.load_image_file("celebrity1.jpg"),
        "celebrity_name_2": face_recognition.load_image_file("celebrity2.jpg"),
        # Add more celebrities as needed
    }

    celebrity_encodings = {name: face_recognition.face_encodings(image)[0] for name, image in celebrities.items()}
    return celebrity_encodings

# Find the celebrity match
def find_celebrity_match(uploaded_face_encoding, celebrity_encodings):
    # Compare the uploaded face encoding with celebrity encodings
    best_match_name = None
    best_match_score = 0

    for celebrity, celebrity_encoding in celebrity_encodings.items():
        results = face_recognition.compare_faces([celebrity_encoding], uploaded_face_encoding)
        face_distance = face_recognition.face_distance([celebrity_encoding], uploaded_face_encoding)[0]

        if results[0]:
            match_score = 1 - face_distance
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_name = celebrity

    return best_match_name, best_match_score

# Generate a funny caption based on the celebrity name
def generate_funny_caption(celebrity_name):
    return f"You look just like a younger version of {celebrity_name}! 😂"
