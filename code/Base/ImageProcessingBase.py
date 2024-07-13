import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from typing import List, Tuple, Any
from IPython.display import display
import os
import pandas as pd


class ImageProcessingBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def process_dataset(self, path: str) -> List[Image.Image]:
        raise NotImplementedError

    # def process_image(self, image: Image.Image) -> List[Image.Image]:
    #     raise NotImplementedError
