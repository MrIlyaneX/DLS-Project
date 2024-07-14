from code.DetectionCut import DetectionCut
import pandas as pd

df = pd.DataFrame(columns=["Original_image", "Component", "Method", "Component_size", "Image_size"])


df = DetectionCut.create_test_dataset(df)
df = DetectionCut.create_test_dataset(df, split='validation')

df.to_csv('test_images.csv', index=False)

