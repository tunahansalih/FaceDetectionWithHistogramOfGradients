"""
The method where the histogram of gradients calculation from Images is done. You will
have two methods: extractHogFromImage and extractHogFromRandomCrop. You will
use extractHogFromImage to get positive class features and extractFromRandomCrop to get
negative class features.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

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
    """

    :param magnitude: numpy nd-array containing magnitude of cells
    :param orientation: numpy nd-array containing orientation of cells
    :param bins: integer, number of bins to calculate
    :return: numpy nd-array, containing histogram of oriented gradients for given cell
    """
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

            cell_histogram[former_bin % bins] += current_magnitude * weight_of_former_bin
            cell_histogram[latter_bin % bins] += current_magnitude * weight_of_latter_bin
    return cell_histogram


def getCellHistograms(image, filter_size, stride, bins):
    """

    :param image: numpy nd-array
    :param filter_size: integer
    :param filter_size: integer
    :param stride: integer
    :param bins: integer
    :return: returns histogram features for each cell in the image
    """
    gradient_x = getGradientX(image)
    gradient_y = getGradientY(image)
    magnitude = getMagnitude(gradient_x, gradient_y)
    orientation = getOrientation(gradient_x, gradient_y)

    cells = [[[] for _ in range((image.shape[1] - filter_size) // stride + 1)] for _ in
             range((image.shape[0] - filter_size) // stride + 1)]

    cell_y = 0
    for y in range(0, image.shape[0] - filter_size + 1, stride):
        cell_x = 0
        for x in range(0, image.shape[1] - filter_size + 1, stride):
            cell_histogram = createHistogramFromMagnitudeAndOrientation(
                magnitude[y:y + filter_size, x:x + filter_size],
                orientation[y:y + filter_size, x:x + filter_size],
                bins)
            cells[cell_y][cell_x] = cell_histogram
            cell_x += 1
        cell_y += 1

    return cells


def normalizeBlock(block):
    """
    L2 normalization for blocks
    :param block:
    :return:
    """
    l2_norm = np.sqrt(np.sum([(np.square(cell)) for cell in block]))
    if l2_norm ==0:
        block = [[0 for value in cell] for cell in block]
    else:
        block = [[value / l2_norm for value in cell] for cell in block]
    return block


def getBlocks(cell_histograms, block_size):
    normalized_blocks = []
    for y in range(len(cell_histograms) - block_size + 1):
        for x in range(len(cell_histograms[0]) - block_size + 1):
            block = [cell_histograms[y + i][x + j] for i in range(block_size) for j in range(block_size)]
            normalized_block = normalizeBlock(block)
            normalized_blocks.append(normalized_block)

    return normalized_blocks


def getFeatureVector(blocks):
    features = []
    for block in blocks:
        feature = []
        for cell in block:
            feature.extend(cell)
        features.append(feature)
    return features


def extractHogFromImage(image, filter_size, stride, bins, block_size):
    cell_histograms = getCellHistograms(image, filter_size, stride, bins)
    blocks = getBlocks(cell_histograms, block_size)
    feature_vectors = getFeatureVector(blocks)
    return feature_vectors


def createUniqueCoordinates(width, height, num_of_points):
    points = set()
    while len(points) < num_of_points:
        points.add((random.randrange(0, width), random.randint(0, height)))
    return list(points)


def extractFromRandomCrop(image, filter_size, stride, bins, block_size, random_percentage):
    cell_histograms = getCellHistograms(image, filter_size, stride*8, bins)
    blocks = getBlocks(cell_histograms, block_size)
    feature_vectors = getFeatureVector(blocks)
    return feature_vectors
