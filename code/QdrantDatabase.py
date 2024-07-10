from typing import Dict, List, Any
import numpy as np
from Base.BaseDatabase import DatabaseBase

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


class QdrantDatabase(DatabaseBase):
    clint: QdrantClient

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6333,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.client = QdrantClient(host=host, port=port, **kwargs)

    def __add(
        self,
        vectors: List[np.ndarray],
        payload: Dict,
        collection_name: str,
        **kwargs: Any,
    ):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=payload,
                )
                for idx, vector in enumerate(vectors)
            ],
        )

    def add(
        self,
        vectors: List[np.ndarray],
        **kwargs: Any,
    ):
        self.__add(vectors, **kwargs)

    def __search(self, querry: List[np.ndarray],  **kwargs: Any):
        pass

    def search(self, querry: List[np.ndarray], **kwargs: Any) -> List[Any]:
        return self.__search(querry, **kwargs)
