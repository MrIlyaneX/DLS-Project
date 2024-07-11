from PIL import Image
from typing import List, Any


class EmbedderBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def embed(self, images: List[Image.Image]) -> List:
        raise NotImplemented
