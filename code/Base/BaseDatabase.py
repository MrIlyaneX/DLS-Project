from typing import Any, List

import numpy as np



class DatabaseBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add(self, vectors: List[np.ndarray], **kwargs: Any):
        raise NotImplemented

    def search(self, qurry: List[np.ndarray], **kwargs: Any) -> List[Any]:
        raise NotImplemented
