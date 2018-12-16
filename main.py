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

random_test_image = cv2.imread(os.path.join(FACE_IMAGES_DIRNAME, FACE_IMAGES_FILENAMES[0]), cv2.IMREAD_GRAYSCALE)
random_test_image = random_test_image / 255.0

hog.extractHoggFromImage(random_test_image, filter_height=3, filter_width=3, stride=1)
