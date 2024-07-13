from code.OIv7Loader import OIv7Loader

dataloader = OIv7Loader(["Flower"], 50)
dataset = dataloader.load_data()

print(dataset)
