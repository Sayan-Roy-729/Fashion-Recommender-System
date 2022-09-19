import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras import Sequential
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def build_model():
    # load the pre-trained model
    model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # make the model non-trainable
    model.trainable = False

    # add our top layer
    model = Sequential([
        model,
        GlobalMaxPooling2D(),
    ])
    return model


def extract_features(img_path, model):
    # load the image as PIL object
    img = image.load_img(img_path, target_size=(224, 224))
    # convert the image to numpy array
    img_array = image.img_to_array(img)
    # create a batch with this single image
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # transform the image according to the resnet50 model, (0 centering)
    preprocessed_img = preprocess_input(expanded_img_array)
    # make a prediction (also converting the result 2D to 1D by flattening)
    result = model.predict(preprocessed_img).flatten()
    # normalize the result
    return result / np.linalg.norm(result)


def create_feature_lists(model):
    # create a list of all the images' names as well as their paths available to out dataset
    filenames = []
    for file in os.listdir(os.path.join("fashion-dataset", "images")):
        filenames.append(os.path.join("fashion-dataset", "images", file))

    # create features of every image and store in a list
    feature_list = []
    for file in tqdm(filenames):
        feature_list.append(extract_features(file, model))

    # store this features
    pickle.dump(feature_list, open("features.pkl", "wb"))
    pickle.dump(filenames, open("filenames.pkl", "wb"))


# get the top 5 nearest neighbors to the uploaded image in our dataset
def nearest_neighbors(feature_lists, features):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_lists)

    # find out the neighbors (distance from the main image and the image indexes of the images of our dataset)
    distances, indices = neighbors.kneighbors([features])
    return distances, indices


if __name__ == "__main__":
    model = build_model()
    create_feature_lists(model)


# Challenges:
# 1. If we have large image dataset, in terms of millions --> Annoy library of Spotify
# 2. Deployment of this project
