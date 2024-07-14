from code.OIv7Loader import OIv7Loader


# Download train dataset
# dataloader = OIv7Loader(["Flower"], 500, path='./dataset/')
# dataset = dataloader.download_data(split='train', shuffle=True)
# dataloader.remove_non_rgb_images()


# Download validation dataset
dataloader = OIv7Loader(["Flower"], 100, path='./dataset/')
dataset = dataloader.download_data(split='validation', shuffle=True)
dataloader.remove_non_rgb_images()

print(dataset)
