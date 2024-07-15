import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from typing import List, Any, Dict
import os


class ImageProcessingBase:
    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:
        self.path = path
        self.data_dir = os.path.join(self.path, 'data')
        image_files = os.listdir(self.data_dir)
        self.image_names = [file for file in image_files]
        self.image_ids = [os.path.splitext(file)[0] for file in image_files]

    def __len__(self):
        return

    def process_dataset(self, path: str) -> Dict[str, List[Image.Image]]:
        raise NotImplementedError

    def get_cropped_fragments(self, image_path, index) -> List[Image.Image]:
        raise NotImplementedError
