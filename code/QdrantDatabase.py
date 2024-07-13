from typing import Any, Dict, List

import numpy as np
from Base.BaseDatabase import DatabaseBase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfig, PointStruct, VectorParams


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

    def create_collection(self, collection_name: str, emb_size: int = 768):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID),
            hnsw_config=HnswConfig(
                m=16,  # Number of bi-directional links created for every new element during construction
                ef_construct=250,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
                ef_search=100,  # Size of the dynamic list for the nearest neighbors during search
            ),
        )

    def __add(
        self,
        vectors: List[np.ndarray],
        idx: List[str | int],
        payload: List[Dict],
        collection_name: str,
        **kwargs: Any,
    ):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id,  # int / str
                    vector=vector.tolist(),
                    payload=pld,
                )
                for id, vector, pld in zip(idx, vectors, payload)
            ],
        )

    def add(
        self,
        vectors: List[np.ndarray],
        **kwargs: Any,
    ):
        self.__add(vectors, **kwargs)

    def __search(self, querry: List[np.ndarray], **kwargs: Any):
        pass

    def search(self, querry: List[np.ndarray], **kwargs: Any) -> List[Any]:
        return self.__search(querry, **kwargs)
