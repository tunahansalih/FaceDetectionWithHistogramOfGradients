"""
Runs the classifier using a sliding window approach on the test set.
Using a scale parameter, for each image, object_detect.py runs the object detection algorithm at multiple scales
and performs non-maxima suppression to remove duplicate detections.
"""