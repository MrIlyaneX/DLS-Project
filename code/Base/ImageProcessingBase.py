from typing import Any


class ImageProcessingBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplemented
