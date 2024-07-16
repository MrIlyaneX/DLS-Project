from code.DetectionCut import DetectionCut
from code.WindowSlidingCut import SlidingWindowCut
import pandas as pd
import os

csv_filename = 'test_images.csv'

if os.path.exists(csv_filename):
    os.remove(csv_filename)

df = pd.DataFrame(columns=["Original_image", "Component", "Method", "Component_size", "Image_size"])

detector = DetectionCut()
window = SlidingWindowCut(300, 270)
df = DetectionCut.create_test_dataset_from_train(df)
df = detector.create_test_dataset_from_valid(df)
df = window.create_test_dataset(df, 25)
df = window.create_test_dataset(df, 25, 'validation')

df.to_csv(csv_filename, index=False)

