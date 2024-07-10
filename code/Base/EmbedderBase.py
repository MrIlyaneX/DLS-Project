from PIL.Image import Image
from typing import List, Any

class EmbedderBase:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def embed(images: List[Image]):
        raise NotImplemented
    