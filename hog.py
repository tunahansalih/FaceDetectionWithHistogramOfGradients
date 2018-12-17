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


def normalizeBlock(block):
    return block


def createHistogramFromMagnitudeAndOrientation(magnitude, orientation, bins):
    # print(magnitude)
    # print(orientation)

    bin_size = MAX_DEGREE // bins
    cell_histogram = np.zeros(bins)
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
            cell_histogram[former_bin % bins] += current_magnitude * weight_of_former_bin
            cell_histogram[latter_bin % bins] += current_magnitude * weight_of_latter_bin
    return cell_histogram


def extractHogFromImage(image, filter_width, filter_height, stride, bins):
    gradient_x = getGradientX(image)
    gradient_y = getGradientY(image)
    magnitude = getMagnitude(gradient_x, gradient_y)
    orientation = getOrientation(gradient_x, gradient_y)

    normalized_cells = []
    # plt.imshow(magnitude, cmap="gray")
    # plt.show()
    # plt.imshow(orientation, cmap="gray")
    # plt.show()
    cells = [[[] for _ in range((image.shape[0] - filter_height) // stride)] for _ in
             range((image.shape[1] - filter_width) // stride)]

    cell_y = 0
    for y in range(0, image.shape[0] - filter_height, stride):
        cell_x = 0
        for x in range(0, image.shape[1] - filter_width, stride):
            print(y, x, cell_y, cell_x)
            cell_histogram = createHistogramFromMagnitudeAndOrientation(
                magnitude[y:y + filter_height, x:x + filter_width],
                orientation[y:y + filter_height, x:x + filter_width],
                bins)
            cells[cell_y][cell_x] = cell_histogram
            cell_x += 1
        cell_y += 1

    for y in range(len(cells) - 1):
        for x in range(len(cells[0]) - 1):
            blocks = [cells[y][x], cells[y + 1][x], cells[y][x + 1], cells[y + 1][x + 1]]
            normalized_block = normalizeBlock(blocks)
            normalized_cells.extend(normalized_block)

    return []


def extractFromRandomCrop(image, filter_width, filter_height, stride, bins):
    pass
