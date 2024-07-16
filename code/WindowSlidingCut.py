import cv2
import os
from tqdm import tqdm
from PIL import Image
from typing import Any, List, Tuple, Dict
from .Base.ImageProcessingBase import ImageProcessingBase
import pandas as pd
import random
import shutil


class SlidingWindowCut(ImageProcessingBase):
    def __init__(self, size: int, step: int, path: str = './dataset/train', *args: Any, **kwargs: Any) -> None:
        super().__init__(path, *args, **kwargs)
        self.size = size
        self.step = step

    def get_cropped_fragments(self, image_path, index) -> List[Image.Image]:
        cropped_fragments = []
        if os.path.exists(image_path):
            image = Image.open(image_path)
            cropped_fragments = self.sliding_window(image, self.size, self.step)
        else:
            print("Image not found (WindowSlidingCut.__getitem__)")
        return cropped_fragments

    @staticmethod
    def sliding_window(image: Image.Image, window_size: int, step_size: int) -> List[Image.Image]:
        cropped_images = []
        image_width, image_height = image.size

        for y in range(0, image_height - window_size + 1, step_size):
            for x in range(0, image_width - window_size + 1, step_size):
                box = (x, y, x + window_size, y + window_size)
                cropped_image = image.crop(box)
                cropped_images.append(cropped_image)

        return cropped_images


    def process_dataset(self, path: str = './dataset/train') -> dict[Any, Any]:
        global_component_counter = 0
        components_dict = {}

        def sliding_window(image, step_size, window_size):
            for y in range(0, image.shape[0] - window_size + 1, step_size):
                for x in range(0, image.shape[1] - window_size + 1, step_size):
                    yield x, y, image[y:y + window_size, x:x + window_size]

        def save_components(image_path, window_size, step_size):
            nonlocal global_component_counter
            image = cv2.imread(image_path)
            if image is None:
                print(f"No image found at {image_path}")
                return []
            components = []
            for (x, y, window) in sliding_window(image, step_size, window_size):
                pil_image = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
                components.append(pil_image)
                global_component_counter += 1
            return components

        def process_images_folder(input_folder, window_size, step_size):
            nonlocal components_dict

            for filename in tqdm(os.listdir(input_folder)):
                if filename.lower().endswith('.jpg'):
                    input_image_path = os.path.join(input_folder, filename)
                    components = save_components(input_image_path, window_size, step_size)
                    components_dict[filename] = components

        input_folder = os.path.join(path, 'data')
        process_images_folder(input_folder, self.size, self.step)
        return components_dict

    @staticmethod
    def create_test_dataset(df, num_images=25, split="train"):
        # Ensure the test directories exist
        os.makedirs("./dataset/test/originals", exist_ok=True)
        os.makedirs("./dataset/test/fragments", exist_ok=True)

        # Get list of image files
        data_dir = f"./dataset/{split}/data"
        image_files = os.listdir(data_dir)

        # Select a random subset of images
        selected_images = random.sample(image_files, min(num_images + 10, len(image_files)))

        # Process each selected image
        num_images_added = 0
        for image_name in selected_images:
            # Copy the original image to the 'originals' folder
            src_path = os.path.join(data_dir, image_name)

            # Crop the image based on the detection boxes
            image_id = os.path.splitext(image_name)[0]
            image = Image.open(src_path)

            # Crop image by sliding window and select random cropped fragment
            cropped_images = SlidingWindowCut.sliding_window(image, 300, 270)
            if cropped_images:
                selected_cropped_image = random.choice(cropped_images)

                # Save the cropped image fragment
                fragment_name = f"{image_id}_{random.randint(10000,99999)}.jpg"
                fragment_path = os.path.join("./dataset/test/fragments", fragment_name)
                selected_cropped_image.save(fragment_path)
                dst_path = os.path.join("./dataset/test/originals", image_name)
                shutil.copyfile(src_path, dst_path)

                df = df._append(
                    {
                        "Original_image": image_name,
                        "Component": fragment_name,
                        "Method": f"window_{split}",
                        "Component_size": selected_cropped_image.size,
                        "Image_size": image.size,
                    },
                    ignore_index=True,
                )
                num_images_added += 1

                if num_images_added >= num_images:
                    # print(f"25 done (win {split[:5]})")
                    break
        return df
