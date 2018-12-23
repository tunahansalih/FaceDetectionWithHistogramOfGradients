"""
Runs the classifier using a sliding window approach on the test set.
Using a scale parameter, for each image, object_detect.py runs the object detection algorithm at multiple scales
and performs non-maxima suppression to remove duplicate detections.
"""
import hog
import numpy as np


def nonMaximaSuppression(predicted_coordinates):
    unique_boxes = []
    if len(predicted_coordinates) > 0:
        overlap_threshold = 0.1

        boxes = []
        for box in predicted_coordinates:
            boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[2]])

        boxes = np.array(boxes, dtype=np.float32)
        picks = []
        x1 = boxes[:, 0]
        x2 = boxes[:, 2]
        y1 = boxes[:, 1]
        y2 = boxes[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        indexes = np.argsort(y2)

        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            picks.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / areas[indexes[:last]]

            # delete all indexes from the index list that have
            indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        for box in boxes[picks].astype(np.int):
            unique_boxes.append([box[0], box[1], box[2] - box[0]])

    return unique_boxes


def detectObject(classifier, image, filter_size, stride, bins, block_size, patch_size, patch_stride):
    image = image / 255.0

    hog_feature_coordinates, hog_feature = hog.extractFromRandomCrop(image,
                                                                     filter_size=filter_size,
                                                                     stride=stride,
                                                                     bins=bins,
                                                                     block_size=block_size,
                                                                     patch_size=patch_size,
                                                                     patch_stride=patch_stride,
                                                                     random_percentage=1)

    predictions_probabilities = classifier.predict_proba(hog_feature)
    hog_feature_coordinates = np.array(hog_feature_coordinates)
    detected_face_coordinates = hog_feature_coordinates[
        predictions_probabilities[:, 1] > 0.99]
    detected_face_coordinates = nonMaximaSuppression(detected_face_coordinates)
    return detected_face_coordinates
