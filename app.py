import streamlit as st
from utils import preprocess_image, load_celebrity_data
from model import find_celebrity_match, generate_funny_caption

# Title
st.title("AI Celebrity Look-Alike Finder 🤩")
st.write("Upload a photo, and let AI tell you which celebrity you resemble!")

# Upload section
uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess and match celebrity
    image = preprocess_image(uploaded_file)
    celebrity_name, similarity_score = find_celebrity_match(image)
    caption = generate_funny_caption(celebrity_name, similarity_score)

    # Display the result
    st.subheader(f"You look like: {celebrity_name}")
    st.write(f"Similarity Score: {similarity_score:.2f}")
    st.write(f"Caption: {caption}")
