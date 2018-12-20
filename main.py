"""
The top level script for training and testing your sliding window based object detector.
"""

import hog
import classifier_train
import object_detect
import evaluation
import visualization
from Data import loadTestsGT
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

FACE_IMAGES_DIRNAME = "Data/FaceImages"
NONFACE_IMAGES_DIRNAME = "Data/NonFaceImages"
VALIDATION_IMAGES_DIRNAME = "Data/ValidationSet"

FACE_IMAGES_FILENAMES = os.listdir(FACE_IMAGES_DIRNAME)
NONFACE_IMAGES_FILENAMES = os.listdir(NONFACE_IMAGES_DIRNAME)
VALIDATION_IMAGES_FILENAMES = os.listdir(VALIDATION_IMAGES_DIRNAME)

FILTER_SIZE = 6
STRIDE = 6
BINS = 9
BLOCK_SIZE = 2

random_test_image = cv2.imread(os.path.join(FACE_IMAGES_DIRNAME, FACE_IMAGES_FILENAMES[0]), cv2.IMREAD_GRAYSCALE)
random_test_image = random_test_image / 255.0

positive_hog_feature_vectors = []
for filename in FACE_IMAGES_FILENAMES:
    print(filename)
    image = cv2.imread(os.path.join(FACE_IMAGES_DIRNAME, filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    image = image / 255.0
    positive_hog_feature_vectors.extend(
        hog.extractHogFromImage(image, filter_size=FILTER_SIZE, stride=STRIDE, bins=BINS, block_size=BLOCK_SIZE))

negative_hog_feature_vectors = []
for filename in NONFACE_IMAGES_FILENAMES:
    print(filename)
    image = cv2.imread(os.path.join(NONFACE_IMAGES_DIRNAME, filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    image = image / 255.0
    negative_hog_feature_vectors.extend(
        hog.extractFromRandomCrop(image, filter_size=FILTER_SIZE, stride=STRIDE, bins=BINS, block_size=BLOCK_SIZE, random_percentage=0.5))

labels = np.zeros(len(positive_hog_feature_vectors) + len(negative_hog_feature_vectors))
labels[:len(positive_hog_feature_vectors)] = 1

feature_vectors = positive_hog_feature_vectors + negative_hog_feature_vectors

classifier_train.train_classifier(feature_vectors, labels)