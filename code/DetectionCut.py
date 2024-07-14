from PIL import Image
from typing import List, Tuple, Any, Dict
from .Base.ImageProcessingBase import ImageProcessingBase
import os
import pandas as pd


class DetectionCut(ImageProcessingBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None

    def process_dataset(self, path: str = './dataset/train') -> Dict[str, List[Image.Image]]:
        data_dir = os.path.join(path, 'data')
        labels_dir = os.path.join(path, 'labels')
        detections_file = os.path.join(labels_dir, 'detections.csv')

        image_files = os.listdir(data_dir)
        image_names = [file for file in image_files]
        image_ids = [os.path.splitext(file)[0] for file in image_files]

        detections_df = pd.read_csv(detections_file)

        flower_label_id = '/m/0c9ph5'
        valid_detections = detections_df[(detections_df['ImageID'].isin(image_ids)) &
                                         (detections_df['LabelName'] == flower_label_id)]

        def load_and_crop_images(data_dir, valid_detections):
            cropped_images = {}

            for image_name in image_names:
                image_id = image_name.split('.')[0]
                image_path = os.path.join(data_dir, image_name)
                cropped_images[image_name] = []

                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    image_detections = valid_detections[valid_detections['ImageID'] == image_id]

                    for index, row in image_detections.iterrows():
                        width, height = image.size
                        xmin = int(row['XMin'] * width)
                        xmax = int(row['XMax'] * width)
                        ymin = int(row['YMin'] * height)
                        ymax = int(row['YMax'] * height)

                        # Crop image based on bounding box
                        cropped_image = image.crop((xmin, ymin, xmax, ymax))
                        cropped_images[image_name].append(cropped_image)

            return cropped_images

        cropped_images = load_and_crop_images(data_dir, valid_detections)
        return cropped_images


