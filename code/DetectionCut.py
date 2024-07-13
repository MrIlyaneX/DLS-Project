import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from typing import List, Tuple, Any
from .Base.ImageProcessingBase import ImageProcessingBase
import pandas as pd
import os


class DetectionCut(ImageProcessingBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None

    def process_dataset(self, path: str = './dataset/train') -> List[Image.Image]:
        images_dir = os.path.join(path, 'data')
        labels_path = os.path.join(path, 'labels/detections.csv')

        # Load detections
        detections = pd.read_csv(labels_path)

        def crop_image(image_path, bbox):
            image = Image.open(image_path)
            cropped_image = image.crop(bbox)
            return cropped_image

        def get_cropped_images(detections, images_dir):
            cropped_images = []

            for _, row in detections.iterrows():
                image_id = row['image_id']
                xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']

                image_path = os.path.join(images_dir, f"{image_id}.jpg")

                if os.path.exists(image_path):
                    # Convert normalized coordinates to pixel coordinates
                    with Image.open(image_path) as img:
                        width, height = img.size
                        bbox = (xmin * width, ymin * height, xmax * width, ymax * height)

                    cropped_image = crop_image(image_path, bbox)
                    cropped_images.append(cropped_image)
                else:
                    print(f"Image {image_id} not found")

            return cropped_images

        cropped_images = get_cropped_images(detections, images_dir)

        return cropped_images
