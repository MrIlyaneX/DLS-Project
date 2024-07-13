from typing import Any, List
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from .Base.DataLoader import DataLoader
import pandas as pd


class OIv7Loader(DataLoader):
    num_images: int
    classes: List[str]
    dataset: fo.core.dataset.Dataset

    def __init__(self, classes: List[str], num_images: int, path: str = './dataset', *args: Any, **kwargs: Any) -> None:
        super().__init__(classes, num_images, *args, **kwargs)
        self.dataset = None
        self.dataset_directory = path

    def download_data(self) -> List:
        self.dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            dataset_dir=self.dataset_directory,
            classes=self.classes,
            max_samples=self.num_images,
            # shuffle=True
        )

        return self.dataset

    def load_data(self, path: str = './dataset/train') -> List:
        images_dir = path + '/data'
        labels_path = path + './dataset/train/labels/detections.csv'

        detections = pd.read_csv(labels_path)



