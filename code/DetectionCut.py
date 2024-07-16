from PIL import Image
from typing import List, Tuple, Any, Dict
from .Base.ImageProcessingBase import ImageProcessingBase
import os
import pandas as pd
import random
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt

K = 0.2
FLOWER_CODE = "/m/0c9ph5"
FLOWER_ID = 195


class DetectionCut(ImageProcessingBase):
    def __init__(
        self, path: str = "./dataset/train", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(path, *args, **kwargs)
        self.model = YOLO("yolov8l-oiv7.pt")

        labels_dir = os.path.join(self.path, "labels")
        detections_file = os.path.join(labels_dir, "detections.csv")
        detections_df = pd.read_csv(detections_file)
        self.valid_detections = detections_df[
            (detections_df["ImageID"].isin(self.image_ids))
            & (detections_df["LabelName"] == FLOWER_CODE)
        ]

    def get_cropped_fragments(self, image_path, index) -> List[Image.Image]:
        cropped_fragments = []
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_detections = self.valid_detections[
                self.valid_detections["ImageID"] == self.image_ids[index]
            ]

            for index, row in image_detections.iterrows():
                width, height = image.size
                xmin = int(row["XMin"] * width)
                xmax = int(row["XMax"] * width)
                ymin = int(row["YMin"] * height)
                ymax = int(row["YMax"] * height)

                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_fragments.append(cropped_image)
        else:
            print("Image not found (DetectionCut.__getitem__)")
        return cropped_fragments

    # сreate 2 separate methods to get cropped image: by detections.csv and by yolo model if no detections.csv
    def detect_and_crop_image(self, path: str) -> List[Image.Image]:
        '''
        Detects flower on the unlabeled image and returns list of the images cropped by detected objects.
        :param path: path to image
        :return: list of PIL images
        '''
        result = self.model(path, conf=0.5, verbose=False)
        cropped_images = []

        for detection in result[0].boxes.data:
            class_id = int(detection[5])
            if class_id == FLOWER_ID:
                xmin, ymin, xmax, ymax = map(int, detection[:4])
                image = Image.open(path)
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_images.append(cropped_image)

        return cropped_images

    def process_dataset(
        self, path: str = "./dataset/train"
    ) -> Dict[str, List[Image.Image]]:
        data_dir = os.path.join(path, "data")
        labels_dir = os.path.join(path, "labels")
        detections_file = os.path.join(labels_dir, "detections.csv")

        image_files = os.listdir(data_dir)
        image_names = [file for file in image_files]
        image_ids = [os.path.splitext(file)[0] for file in image_files]

        detections_df = pd.read_csv(detections_file)

        valid_detections = detections_df[
            (detections_df["ImageID"].isin(image_ids))
            & (detections_df["LabelName"] == FLOWER_ID)
        ]

        def load_and_crop_images(data_dir, valid_detections):
            cropped_images = {}

            for image_name in image_names:
                image_id = image_name.split(".")[0]
                image_path = os.path.join(data_dir, image_name)
                cropped_images[image_name] = []

                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    image_detections = valid_detections[
                        valid_detections["ImageID"] == image_id
                    ]

                    for index, row in image_detections.iterrows():
                        width, height = image.size
                        xmin = int(row["XMin"] * width)
                        xmax = int(row["XMax"] * width)
                        ymin = int(row["YMin"] * height)
                        ymax = int(row["YMax"] * height)

                        # Crop image based on bounding box
                        cropped_image = image.crop((xmin, ymin, xmax, ymax))
                        cropped_images[image_name].append(cropped_image)

            return cropped_images

        cropped_images = load_and_crop_images(data_dir, valid_detections)
        return cropped_images

    def create_test_dataset_from_valid(self, df, num_images=25, split="validation"):
        # Ensure the test directories exist
        os.makedirs("./dataset/test/originals", exist_ok=True)
        os.makedirs("./dataset/test/fragments", exist_ok=True)

        # Get list of image files
        data_dir = f"./dataset/{split}/data"
        image_files = os.listdir(data_dir)

        # Select a random subset of images
        selected_images = random.sample(image_files, min(num_images + 20, len(image_files)))

        # Process each selected image
        num_images_added = 0
        for image_name in selected_images:
            # Copy the original image to the 'originals' folder
            src_path = os.path.join(data_dir, image_name)

            # Crop the image based on the detection boxes
            image_id = os.path.splitext(image_name)[0]
            image = Image.open(src_path)

            # Detect and crop flowers from the image
            cropped_images = self.detect_and_crop_image(src_path)
            if cropped_images:
                selected_cropped_image = random.choice(cropped_images)

                # Save the cropped image fragments
                fragment_name = f"{image_id}_{random.randint(10000,99999)}.jpg"
                fragment_path = os.path.join("./dataset/test/fragments", fragment_name)
                selected_cropped_image.save(fragment_path)
                dst_path = os.path.join("./dataset/test/originals", image_name)
                shutil.copyfile(src_path, dst_path)

                df = df._append(
                    {
                        "Original_image": image_name,
                        "Component": fragment_name,
                        "Method": f"detection_{split}",
                        "Component_size": selected_cropped_image.size,
                        "Image_size": image.size,
                    },
                    ignore_index=True,
                )
                num_images_added += 1

                if num_images_added >= num_images:
                    # print("25 done (det valid)")
                    break
        return df

    @staticmethod
    def create_test_dataset_from_train(df, num_images=25, split="train"):
        # Ensure the test directories exist
        os.makedirs("./dataset/test/originals", exist_ok=True)
        os.makedirs("./dataset/test/fragments", exist_ok=True)

        # Get list of image files
        data_dir = f"./dataset/{split}/data"
        image_files = os.listdir(data_dir)

        # Select a random subset of images
        selected_images = random.sample(image_files, min(num_images + 10, len(image_files)))

        # Load the detections CSV
        detections_df = pd.read_csv(f"./dataset/{split}/labels/detections.csv")
        valid_detections = detections_df[detections_df["LabelName"] == FLOWER_CODE]

        # Process each selected image
        num_images_added = 0
        for image_name in selected_images:
            # Copy the original image to the 'originals' folder
            src_path = os.path.join(data_dir, image_name)

            # Crop the image based on the detection boxes
            image_id = os.path.splitext(image_name)[0]
            image = Image.open(src_path)
            width, height = image.size

            image_detections = valid_detections[valid_detections["ImageID"] == image_id]
            detections_list = list(image_detections.iterrows())
            if detections_list:
                selected_image_detection = random.choice(detections_list)
                index, row = selected_image_detection

                xmin = int(row["XMin"] * width)
                xmax = int(row["XMax"] * width)
                ymin = int(row["YMin"] * height)
                ymax = int(row["YMax"] * height)

                # Calculate random shifts
                shift_x = int((xmax - xmin) * K)
                shift_y = int((ymax - ymin) * K)

                xmin_shifted = max(0, xmin + random.randint(-shift_x, shift_x))
                xmax_shifted = min(width, xmax + random.randint(-shift_x, shift_x))
                ymin_shifted = max(0, ymin + random.randint(-shift_y, shift_y))
                ymax_shifted = min(height, ymax + random.randint(-shift_y, shift_y))

                cropped_image = image.crop(
                    (xmin_shifted, ymin_shifted, xmax_shifted, ymax_shifted)
                )

                # Save the cropped image fragment
                fragment_name = f"{image_id}_{random.randint(10000,99999)}.jpg"
                fragment_path = os.path.join("./dataset/test/fragments", fragment_name)
                cropped_image.save(fragment_path)
                dst_path = os.path.join("./dataset/test/originals", image_name)
                shutil.copyfile(src_path, dst_path)

                df = df._append(
                    {
                        "Original_image": image_name,
                        "Component": fragment_name,
                        "Method": f"detection_{split}",
                        "Component_size": cropped_image.size,
                        "Image_size": image.size,
                    },
                    ignore_index=True,
                )
                num_images_added += 1

                if num_images_added >= num_images:
                    # print("25 done (det train)")
                    break

        return df
