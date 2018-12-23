"""
Computes average precision and mean Intersection over Union score for each image and the entire validation dataset.
"""
import numpy as np


def intersection_union(box1, box2):
    x_min1 = box1[0]
    y_min1 = box1[1]
    x_max1 = box1[0] + box1[2]
    y_max1 = box1[1] + box1[2]

    x_min2 = box2[1]
    y_min2 = box2[0]
    x_max2 = box2[3]
    y_max2 = box2[2]

    xx1 = np.max([x_min1, x_min2])
    yy1 = np.max([y_min1, y_min2])
    xx2 = np.min([x_max1, x_max2])
    yy2 = np.min([y_max1, y_max2])

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    union_area = area1 + area2 - inter_area

    return inter_area, union_area


def intersection_over_union(detected_coordinates, actual_coordinates):
    intersection_area = 0
    union_area = 0
    for detected in detected_coordinates:
        for actual in actual_coordinates:
            inter, union = intersection_union(detected, actual)
            if inter > 0:
                intersection_area += inter
                union_area += union
    if union_area > 0:
        return (1.0 * intersection_area) / (1.0 * union_area)
    else:
        return 0
