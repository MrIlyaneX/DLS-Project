from typing import List, Any


class DataLoader:
    def __init__(self, classes: List[str], num_images: int, *args: Any, **kwargs: Any) -> None:
        self.classes = classes
        self.num_images = num_images

    def download_data(self) -> List:
        raise NotImplementedError

    def load_data(self, path: str) -> List:
        raise NotImplementedError
