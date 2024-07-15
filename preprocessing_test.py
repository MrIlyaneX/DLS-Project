from code.DetectionCut import DetectionCut
from code.WindowSlidingCut import SlidingWindowCut
from IPython.display import display
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# # Version 1
# detector = DetectionCut()
# imgs = detector.process_dataset()
#
# print(len(imgs))
# print(len(imgs['0000b366aaf9672a.jpg']))
# print(len(imgs['0000ef4409880196.jpg']))
#
#
def show_images(images, titles=None, columns=1, figsize=(15, 10)):

    plt.figure(figsize=figsize)
    num_images = len(images)
    rows = (num_images + columns - 1) // columns
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.axis('off')
        if titles:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()


# show_images(imgs['0000ef4409880196.jpg'])

# Version 2

# detector = DetectionCut('./dataset/validation')
# print(detector.__len__())
# print(detector[6]['269191f83be6c348.jpg'])
# show_images(detector[6]['269191f83be6c348.jpg'])

# window = SlidingWindowCut(path='./dataset/validation', step=270, size=300)
# print(len(window))
# print(window[6])
# show_images(window[6]['269191f83be6c348.jpg'])


