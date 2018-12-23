"""
Visualizes detections on each image and displays / saves the entire validation set in a loop.
On the same image, draw ground truth bounding boxes as red and predicted bounding boxes as green.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def visualizePredicted(image, predicted_coordinates, actual_coordinates, name):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image, cmap="gray")
    for i, coordinate in enumerate(predicted_coordinates):
        # Create a Rectangle patch
        rect = patches.Rectangle((coordinate[1], coordinate[0]),
                                 coordinate[2], coordinate[2],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    for i, coordinate in enumerate(actual_coordinates):
        # Create a Rectangle patch
        rect = patches.Rectangle((coordinate[0], coordinate[1]),
                                 coordinate[2] - coordinate[0], coordinate[3] - coordinate[1],
                                 linewidth=1, edgecolor='g',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.title(name)
    plt.imsave(os.path.join("images", name))
    plt.show()
