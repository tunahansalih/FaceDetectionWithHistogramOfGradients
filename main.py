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
from joblib import dump, load
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filter_size', type=int, required=True, help='Filter Size')
ap.add_argument('-b', '--bins', type=int, required=True, help='Number of bins for Hog feature')
ap.add_argument('-l', '--block-size', required=True, type=int, help='Block size for normalization')
ap.add_argument('-p', '--patch-size', required=True, type=int, help='Patch size')
ap.add_argument('-s', '--patch-stride', type=int, required=True, help='Patch Stride')
ap.add_argument('-m', '--trained-model', help="Trained SVM model")
args = vars(ap.parse_args())

FACE_IMAGES_DIRNAME = "Data/FaceImages"
NONFACE_IMAGES_DIRNAME = "Data/NonFaceImages"
VALIDATION_IMAGES_DIRNAME = "Data/ValidationSet"

FACE_IMAGES_FILENAMES = os.listdir(FACE_IMAGES_DIRNAME)
NONFACE_IMAGES_FILENAMES = os.listdir(NONFACE_IMAGES_DIRNAME)
VALIDATION_IMAGES_FILENAMES = os.listdir(VALIDATION_IMAGES_DIRNAME)

FILTER_SIZE = args.get("filter_size", 6)
STRIDE = args.get("filter_size", 6)
BINS = args.get("bins", 9)
BLOCK_SIZE = args.get("block_size", 3)
RANDOM_CROP_PERCENTAGE = 0.1
PATCH_SIZE = args.get("patch_size", 32)
PATCH_STRIDE = args.get("patch_stride", 4)

model_name = f"f_{FILTER_SIZE}_b_{BINS}_l_{BLOCK_SIZE}_p_{PATCH_SIZE}_s_{STRIDE}"

if not os.path.exists(f"svm_classifier_{model_name}.joblib") and args["trained_model"] is None:
    positive_hog_faature_coordinates = []
    positive_hog_feature_vectors = []
    for filename in FACE_IMAGES_FILENAMES:
        print(filename)
        image = cv2.imread(os.path.join(FACE_IMAGES_DIRNAME, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = image / 255.0
        feature_vector_coordinates, feature_vector = hog.extractHogFromImage(image, filter_size=FILTER_SIZE,
                                                                             stride=STRIDE,
                                                                             bins=BINS,
                                                                             block_size=BLOCK_SIZE)
        positive_hog_faature_coordinates.extend(feature_vector_coordinates)
        positive_hog_feature_vectors.append(feature_vector)

    negative_hog_feature_coordinates = []
    negative_hog_feature_vectors = []
    for filename in NONFACE_IMAGES_FILENAMES:
        print(filename)
        image = cv2.imread(os.path.join(NONFACE_IMAGES_DIRNAME, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = image / 255.0
        feature_vector_coordinates, feature_vector = hog.extractFromRandomCrop(image, filter_size=FILTER_SIZE,
                                                                               stride=STRIDE, bins=BINS,
                                                                               block_size=BLOCK_SIZE,
                                                                               patch_size=PATCH_SIZE,
                                                                               patch_stride=PATCH_STRIDE,
                                                                               random_percentage=RANDOM_CROP_PERCENTAGE)
        negative_hog_feature_coordinates.extend(feature_vector_coordinates)
        negative_hog_feature_vectors.extend(feature_vector)

    labels = np.zeros(len(positive_hog_feature_vectors) + len(negative_hog_feature_vectors))
    labels[:len(positive_hog_feature_vectors)] = 1

    feature_vectors = positive_hog_feature_vectors + negative_hog_feature_vectors
    print("[INFO] Training SVM...")
    svm_classifier = classifier_train.train_classifier(feature_vectors, labels)
    print("[INFO] Pickling SVN Classifier")
    dump(svm_classifier, f"svm_classifier_{model_name}.joblib")
elif args["trained_model"] is not None:
    svm_classifier = load(args["trained_model"])
else:
    svm_classifier = load(f"svm_classifier_{model_name}.joblib", "r")

actual_coordinates = {}
actual_coordinates_list = loadTestsGT.loadTestsGT()
for filename in VALIDATION_IMAGES_FILENAMES:
    actual_coordinates[filename] = []

for c in actual_coordinates_list:
    actual_coordinates[c[0]].append([c[1], c[2], c[3], c[4]])

f = open(f"{model_name}_results.txt", "w")

total_intersection_over_union = 0
for filename in VALIDATION_IMAGES_FILENAMES:
    print(filename)

    image = cv2.imread(os.path.join(VALIDATION_IMAGES_DIRNAME, filename),
                       cv2.IMREAD_GRAYSCALE)

    detected_coordinates = object_detect.detectObject(classifier=svm_classifier,
                                                      image=image,
                                                      filter_size=FILTER_SIZE,
                                                      stride=STRIDE,
                                                      bins=BINS,
                                                      block_size=BLOCK_SIZE,
                                                      patch_size=PATCH_SIZE,
                                                      patch_stride=PATCH_STRIDE)

    intersection_over_union = evaluation.intersection_over_union(detected_coordinates=detected_coordinates,
                                                                 actual_coordinates=actual_coordinates[filename])
    total_intersection_over_union += intersection_over_union
    print(f"{filename} intersection over union: {intersection_over_union}")
    visualization.visualizePredicted(image, detected_coordinates, actual_coordinates[filename],
                                     f"{model_name}_{filename}")

    f.write(f"{filename} intersection over union: {intersection_over_union}")

f.write(f"Mean Intersection Over Union: {total_intersection_over_union/len(VALIDATION_IMAGES_FILENAMES)}")
f.close()
