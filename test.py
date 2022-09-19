import os
import cv2
import pickle

import utilities

feature_lists = pickle.load(open("features.pkl", "rb"))
image_paths = pickle.load(open("filenames.pkl", "rb"))

model = utilities.build_model()

image_path = os.path.join("test-images", "shoe.jpg")
normalized_result = utilities.extract_features(image_path, model)

# find top 6 nearest neighbors among the images
distances, indices = utilities.nearest_neighbors(feature_lists, normalized_result)
# get the top 5 matched file paths
for file in indices[0]:
    temp_img = cv2.imread(image_paths[file])
    cv2.imshow("output", cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
