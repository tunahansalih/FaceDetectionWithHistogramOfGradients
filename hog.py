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

    # cells = [[[] for _ in range((image.shape[1] - filter_size) // stride + 1)] for _ in
    #          range((image.shape[0] - filter_size) // stride + 1)]
    cells = {}
    for y in range(0, image.shape[0] - filter_size + 1, stride):
        for x in range(0, image.shape[1] - filter_size + 1, stride):
            cell_histogram = createHistogramFromMagnitudeAndOrientation(
                magnitude[y:y + filter_size, x:x + filter_size],
                orientation[y:y + filter_size, x:x + filter_size],
                bins)
            # cells[cell_y][cell_x] = cell_histogram
            cells[(y, x)] = cell_histogram
    return cells


def normalizeBlock(block):
    """
    L2 normalization for blocks
    :param block:
    :return:
    """
    l2_norm = np.sqrt(np.sum([(np.square(cell)) for cell in block]))
    if l2_norm == 0:
        block = [[0 for _ in cell] for cell in block]
    else:
        block = [[value / l2_norm for value in cell] for cell in block]
    return block


def getBlocks(cell_histograms, block_size, stride):
    # normalized_blocks = []
    # for y in range(len(cell_histograms) - block_size + 1):
    #     for x in range(len(cell_histograms[0]) - block_size + 1):
    #         block = [cell_histograms[y + i][x + j] for i in range(block_size) for j in range(block_size)]
    #         normalized_block = normalizeBlock(block)
    #         normalized_blocks.append(normalized_block)
    normalized_block = {}

    block_coordinates = np.array(list(cell_histograms.keys()))
    (min_y, min_x), (max_y, max_x) = np.min(block_coordinates, axis=0), np.max(block_coordinates, axis=0)
    for y in range(min_y, max_y - (block_size - 1) * stride + 1, stride):
        for x in range(min_x, max_x - (block_size - 1) * stride + 1, stride):
            block = [cell_histograms[(y + i * stride, x + j * stride)]
                     for i in range(block_size)
                     for j in range(block_size)]
            normalized_block[(y, x)] = normalizeBlock(block)
    return normalized_block


def getFeatureVector(blocks):
    feature = []
    for (_, _), block in blocks.items():
        for cell in block:
            feature.extend(cell)

    return feature


def extractHogFromImage(image, filter_size, stride, bins, block_size):
    cell_histograms = getCellHistograms(image, filter_size, stride, bins)
    blocks = getBlocks(cell_histograms, block_size, stride)
    feature_vector = getFeatureVector(blocks)
    return [(0, 0)], feature_vector


def extractFromRandomCrop(image, filter_size, stride, bins, block_size, patch_size, patch_stride, random_percentage):
    scale = 1.0
    feature_vectors = {}
    while image.shape[0] * scale > patch_size and image.shape[1] * scale > patch_size:
        image = cv2.resize(image, None, fx=scale, fy=scale)
        for y in range(0, image.shape[0] - patch_size + 1, patch_stride):
            for x in range(0, image.shape[1] - patch_size + 1, patch_stride):
                if np.random.random() < random_percentage:
                    img_patch = image[y:y + patch_size, x:x + patch_size]
                    img_patch = cv2.resize(img_patch, (36, 36))
                    cell_histograms = getCellHistograms(img_patch, filter_size, stride, bins)
                    blocks = getBlocks(cell_histograms, block_size, stride)
                    feature_vectors[(int(y / scale), int(x / scale), int(patch_size / scale))] = getFeatureVector(
                        blocks)
        scale *= (2 / 3)
    return list(feature_vectors.keys()), list(feature_vectors.values())
