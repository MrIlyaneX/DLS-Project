from typing import List, Any


class DataLoader:
    def __init__(self, classes: List[str], num_images: int, *args: Any, **kwargs: Any) -> None:
        self.classes = classes
        self.num_images = num_images

    def load_data(self) -> List:
        raise NotImplementedError
