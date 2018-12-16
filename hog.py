"""
The method where the histogram of gradients calculation from Images is done. You will
have two methods: extractHogFromImage and extractHogFromRandomCrop. You will
use extractHogFromImage to get positive class features and extractFromRandomCrop to get
negative class features.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def extractHoggFromImage(image, filter_width, filter_height, stride):
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    edges_x = cv2.filter2D(image, -1, kernelx)
    edges_y = cv2.filter2D(image, -1, kernely)

    plt.imshow(edges_x, cmap='gray')
    plt.show()
    plt.imshow(edges_y, cmap='gray')
    plt.show()



