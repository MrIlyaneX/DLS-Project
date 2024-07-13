from typing import Any, Dict, List

import numpy as np
from qdrant_client.http.models.models import ScoredPoint

from .Base.BaseDatabase import DatabaseBase
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
        self.client = QdrantClient(location=":memory:")
        #self.client = QdrantClient(host=host, port=port, **kwargs)

    def create_collection(
        self,
        collection_name: str,
        m: int = 16,
        ef_construct: int = 250,
        ef_search: int = 100,
        emb_size: int = 768,
    ) -> None:
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID),
                hnsw_config=HnswConfig(
                    m=m,  # Number of bi-directional links created for every new element during construction
                    ef_construct=ef_construct,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
                    ef_search=ef_search,  # Size of the dynamic list for the nearest neighbors during search
                    full_scan_threshold=10000,
                ),
            )

    def __add(
        self,
        vectors: List[np.ndarray],
        idx: List[str | int],
        payload: List[Dict],
        collection_name: str,
        **kwargs: Any,
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id,  # int / str
                    vector=vector,
                    payload=pld,
                )
                for i, id, vector, pld in enumerate(zip(idx, vectors, payload))
            ],
        )

    def add(
        self,
        vectors: List[np.ndarray],
        **kwargs: Any,
    ) -> None:
        self.__add(vectors, **kwargs)

    def __search(
        self,
        querry: List[np.ndarray],
        collection_name: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[List[ScoredPoint]]:
        return [
            self.client.search(
                collection_name=collection_name,
                query_vector=q,
                limit=limit,
                with_vectors=False,
            )
            for q in querry
        ]

    def search(self, querry: List[np.ndarray], **kwargs: Any) -> List[Any]:
        return self.__search(querry, **kwargs)
