from code.OIv7Loader import OIv7Loader
from code.DetectionCut import DetectionCut
from IPython.display import display
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

detectionCutting = DetectionCut()
imgs = detectionCutting.process_dataset()

print(len(imgs))
print(len(imgs['0000b366aaf9672a.jpg']))
print(len(imgs['0000ef4409880196.jpg']))


def show_images(images, titles=None, columns=5, figsize=(15, 10)):
    """
    Display a list of images in a grid format.

    Parameters:
        images (list): List of PIL.Image objects.
        titles (list, optional): List of titles to display below each image.
        columns (int, optional): Number of columns in the grid.
        figsize (tuple, optional): Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    num_images = len(images)
    rows = (num_images + columns - 1) // columns  # Round up to the nearest whole number of rows

    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.axis('off')
        if titles:
            plt.title(titles[i])

    plt.tight_layout()
    plt.show()


show_images(imgs['0000b366aaf9672a.jpg'])



