import os
import pickle
from PIL import Image
import streamlit as st

import utilities

# load all the features & image paths
feature_lists = pickle.load(open("features.pkl", "rb"))
image_paths = pickle.load(open("filenames.pkl", "rb"))

st.title("Fashion Recommender System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        return 0


def feature_extraction(img_path):
    # load the pre-trained model
    model = utilities.build_model()
    return utilities.extract_features(img_path, model)


def recommend(features, feature_lists):
    _, indices = utilities.nearest_neighbors(feature_lists, features)
    return indices[0]  # indices is a 2D array


# file upload and save it
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # file has been uploaded, display that
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extraction
        image_path = os.path.join("uploads", uploaded_file.name)
        features = feature_extraction(image_path)
        # recommendation
        indices = recommend(features, feature_lists)
        # display the recommended images
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(image_paths[indices[0]])
        with col2:
            st.image(image_paths[indices[1]])
        with col3:
            st.image(image_paths[indices[2]])
        with col4:
            st.image(image_paths[indices[3]])
        with col5:
            st.image(image_paths[indices[4]])

        # delete the uploaded image file
        os.remove(image_path)
    else:
        st.header("Some error occurred to upload the file")
