from code.DetectionCut import DetectionCut
import pandas as pd

df = pd.DataFrame(columns=["Original_image", "Component", "Method", "Component_size", "Image_size"])

detector = DetectionCut()
df = detector.create_test_dataset_from_train(df)
df = detector.create_test_dataset_from_valid(df)

df.to_csv('test_images.csv', index=False)

