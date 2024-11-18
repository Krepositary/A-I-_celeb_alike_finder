import streamlit as st
import face_recognition
from utils import preprocess_image, load_celebrity_data
from model import find_celebrity_match, generate_funny_caption

# Display the title of the app
st.title("AI Celebrity Look-Alike Finder")

# File uploader widget
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)

    # Find the faces in the uploaded image
    uploaded_image_encoding = face_recognition.face_encodings(image)

    if len(uploaded_image_encoding) == 0:
        st.write("No faces detected in the uploaded image.")
    else:
        # Load celebrity images
        celebrity_images = load_celebrity_data()

        # Find the celebrity match
        best_match_name, best_match_score = find_celebrity_match(uploaded_image_encoding, celebrity_images)

        # Generate funny caption based on the match
        caption = generate_funny_caption(best_match_name)

        # Display the result
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write(f"You resemble {best_match_name} with a score of {best_match_score:.2f}.")
        st.write(caption)

