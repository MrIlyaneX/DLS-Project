from PIL import Image
from typing import List, Tuple, Any, Dict
from .Base.ImageProcessingBase import ImageProcessingBase
import os
import pandas as pd
import random
import shutil
from ultralytics import YOLO

K = 0.2

class DetectionCut(ImageProcessingBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = YOLO('yolov8m.pt')


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

    def detect_and_crop_image(self, path):
        results = model(source_path, conf=0.5, data=data_path, save=True)

    @staticmethod
    def create_test_dataset(df, num_images=25, split='train'):
        # Ensure the test directories exist
        os.makedirs('./dataset/test/originals', exist_ok=True)
        os.makedirs('./dataset/test/fragments', exist_ok=True)

        # Get list of image files
        data_dir = f'./dataset/{split}/data'
        image_files = os.listdir(data_dir)

        # Select a random subset of images
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        # Load the detections CSV
        detections_df = pd.read_csv(f'./dataset/{split}/labels/detections.csv')
        flower_label_id = '/m/0c9ph5'
        valid_detections = detections_df[detections_df['LabelName'] == flower_label_id]

        # Process each selected image
        for image_name in selected_images:
            # Copy the original image to the 'originals' folder
            src_path = os.path.join(data_dir, image_name)
            dst_path = os.path.join('./dataset/test/originals', image_name)
            shutil.copyfile(src_path, dst_path)

            # Crop the image based on the detection boxes
            image_id = os.path.splitext(image_name)[0]
            image = Image.open(src_path)
            width, height = image.size

            image_detections = valid_detections[valid_detections['ImageID'] == image_id]

            for index, row in image_detections.iterrows():
                xmin = int(row['XMin'] * width)
                xmax = int(row['XMax'] * width)
                ymin = int(row['YMin'] * height)
                ymax = int(row['YMax'] * height)

                # Calculate random shifts
                shift_x = int((xmax - xmin) * K)
                shift_y = int((ymax - ymin) * K)

                xmin_shifted = max(0, xmin + random.randint(-shift_x, shift_x))
                xmax_shifted = min(width, xmax + random.randint(-shift_x, shift_x))
                ymin_shifted = max(0, ymin + random.randint(-shift_y, shift_y))
                ymax_shifted = min(height, ymax + random.randint(-shift_y, shift_y))

                cropped_image = image.crop((xmin_shifted, ymin_shifted, xmax_shifted, ymax_shifted))

                # Save the cropped image fragment
                fragment_name = f"{image_id}_{index}.jpg"
                fragment_path = os.path.join('./dataset/test/fragments', fragment_name)
                cropped_image.save(fragment_path)

                df = df._append({
                    "Original_image": image_name,
                    "Component": fragment_name,
                    "Method": "detection",
                    "Component_size": cropped_image.size,
                    "Image_size": image.size
                }, ignore_index=True)

        return df


