"""
The method where the histogram of gradients calculation from Images is done. You will
have two methods: extractHogFromImage and extractHogFromRandomCrop. You will
use extractHogFromImage to get positive class features and extractFromRandomCrop to get
negative class features.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

MAX_DEGREE = 180


def getGradientX(image):
    """

    :param image: image matrix
    :return: numpy array with the same shape of image
    """
    gradient_x = np.zeros_like(image)
    gradient_x[:, 1:-1] = -image[:, :-2] + image[:, 2:]
    gradient_x[:, 0] = -image[:, 0] + image[:, 1]
    gradient_x[:, -1] = -image[:, -2] + image[:, -1]

    return gradient_x


def getGradientY(image):
    """

    :param image: image matrix
    :return: numpy array with the same shape of image
    """
    gradient_y = np.zeros_like(image)
    gradient_y[1:-1, :] = image[:-2, :] - image[2:, :]
    gradient_y[0, :] = image[0, :] - image[1, :]
    gradient_y[-1, :] = image[-2, :] - image[-1, :]

    return gradient_y


def getMagnitude(gradientX, gradientY):
    """

    :param gradientX: numpy nd-array
    :param gradientY: numpy nd-array
    :return: numpy nd-array with the same shape of gradientX and gradientY
    """
    return np.sqrt(np.add(np.square(gradientX), np.square(gradientY)))


def getOrientation(gradientX, gradientY):
    """
    :param gradientX: numpy nd-array
    :param gradientY: numpy nd-array
    :return: numpy nd-array with the same shape of gradientX and gradientY
    """
    return np.arctan2(gradientY, gradientX) * 180 / np.pi % 180


def createHistogramFromMagnitudeAndOrientation(magnitude, orientation, bins):
    # print(magnitude)
    # print(orientation)

    bin_size = MAX_DEGREE // bins
    bin_array = np.zeros(bins)
    for y in range(orientation.shape[0]):
        for x in range(orientation.shape[1]):
            current_orientation = orientation[y, x]
            current_magnitude = magnitude[y, x]
            # Interpolate magnitude over neighbor bins
            former_bin = int(current_orientation // bin_size)
            latter_bin = former_bin + 1
            weight_of_former_bin = 1 + former_bin - current_orientation / bin_size
            weight_of_latter_bin = 1 - weight_of_former_bin

            # print(current_orientation, former_bin, weight_of_former_bin)
            bin_array[former_bin % bins] += current_magnitude * weight_of_former_bin
            bin_array[latter_bin % bins] += current_magnitude * weight_of_latter_bin
    return bin_array


def extractHogFromImage(image, filter_width, filter_height, stride, bins):
    gradient_x = getGradientX(image)
    gradient_y = getGradientY(image)
    magnitude = getMagnitude(gradient_x, gradient_y)
    orientation = getOrientation(gradient_x, gradient_y)

    # plt.imshow(magnitude, cmap="gray")
    # plt.show()
    # plt.imshow(orientation, cmap="gray")
    # plt.show()

    for y in range(0, image.shape[0] - filter_width, stride):
        for x in range(0, image.shape[1] - filter_height, stride):
            bin_array = createHistogramFromMagnitudeAndOrientation(magnitude[y:y + filter_width, x:x + filter_height],
                                                                   orientation[y:y + filter_width, x:x + filter_height],
                                                                   bins)

            print(bin_array)
