from code.OIv7Loader import OIv7Loader
from code.DetectionCut import DetectionCut
from IPython.display import display

# dataloader = OIv7Loader(["Flower"], 50)
# dataset = dataloader.load_data()

# print(dataset)

processing = DetectionCut()
cropped_images = processing.process_dataset()

print(cropped_images[0])
display(cropped_images[0])
