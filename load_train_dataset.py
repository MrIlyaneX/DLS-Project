from code.OIv7Loader import OIv7Loader
from code.DetectionCut import DetectionCut
from IPython.display import display

dataloader = OIv7Loader(["Flower"], 700, path='./dataset/train/')
dataset = dataloader.download_data()
dataloader.remove_non_rgb_images()

print(dataset)

# processing = DetectionCut()
# cropped_images = processing.process_dataset()
#
# print(cropped_images[0])
# display(cropped_images[0])


# s = 'dkvnfjln.jpg'
# print(s.split('.')[0])
