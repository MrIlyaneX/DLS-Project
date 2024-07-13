import cv2
import os
from tqdm import tqdm
from PIL import Image
from typing import Any, List, Tuple, Dict
from .Base.ImageProcessingBase import ImageProcessingBase


class SlidingWindowCut(ImageProcessingBase):
    def __init__(self, size: int, step: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        self.step = step

    def process_dataset(self, path: str = './dataset/train') -> dict[Any, Any]:
        global_component_counter = 0
        components_dict = {}

        def sliding_window(image, step_size, window_size):
            for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
                for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                    yield x, y, image[y:y + window_size[1], x:x + window_size[0]]

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

        input_folder = os.path.join(path, 'images')
        process_images_folder(input_folder, self.size, self.step)
        return components_dict