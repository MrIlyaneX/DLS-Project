import qdrant_client
import qdrant_client.conversions
import qdrant_client.conversions.common_types
from sympy import im
from tqdm import tqdm
from code.NomicEmbedder import NomicEmbedder
from code.QdrantDatabase import QdrantDatabase
from code.WindowSlidingCut import SlidingWindowCut
from code.DetectionCut import DetectionCut

from PIL import Image

from qdrant_client import models

import numpy as np
import pandas as pd

from metrics import get_metrics

collection_name: str = "image_vector_store_v1"
emb_size: int = 768
k: int = 10
image_directory: str = "./dataset/train/"

b_size = 500


def initialize() -> tuple[NomicEmbedder, QdrantDatabase]:
    embedder = NomicEmbedder(device="mps", batch_size=4)
    database = QdrantDatabase(host="localhost", port=6333)

    sliding_window = SlidingWindowCut(size=300, step=270)
    cropper = DetectionCut()

    database.create_collection(
        collection_name=collection_name,
        emb_size=emb_size,
        m=16,
        ef_construct=250,
        ef_search=100,
    )

    return embedder, database, [sliding_window, cropper]


def dimentionality_reduction():
    from sklearn.random_projection import GaussianRandomProjection

    embedder, database, preprocessors = initialize()

    images = [
        preprocessor.process_dataset(image_directory) for preprocessor in preprocessors
    ]

    image_names = images[0].keys()
    with open("./data/embeddings.npy", "wb") as f:
        for image_name in tqdm(image_names):
            np.save(
                f,
                np.array(
                    embedder.embed(
                        [
                            img
                            for i in range(len(images))
                            for img in images[i][image_name]
                        ]
                    )
                ),
            )


def process_dataset() -> None:
    embedder, database, preprocessors = initialize()

    images = [
        preprocessor.process_dataset(image_directory) for preprocessor in preprocessors
    ]

    image_names = images[0].keys()
    counter = 0
    for image_name in tqdm(image_names):
        image_embeddings = embedder.embed(
            [img for i in range(len(images)) for img in images[i][image_name]]
        )
        idx = range(counter, counter + len(image_embeddings))
        counter += len(image_embeddings)

        database.add(
            vectors=image_embeddings,
            idx=[i for i in idx],
            payload=[{"source": image_name} for _ in range(len(image_embeddings))],
            collection_name=collection_name,
        )


def search_test():
    import os

    embedder = NomicEmbedder(device="mps", batch_size=4)
    database = QdrantDatabase(host="localhost", port=6333)

    image_test_data_csv = pd.read_csv("test_images.csv", index_col=False)

    images = [
        Image.open("./dataset/test/fragments/" + img_path)
        for img_path in os.listdir("./dataset/test/fragments")
    ]

    query_embeddings = embedder.embed(images)

    results = database.search(
        querry=query_embeddings,
        collection_name=collection_name,
        limit=k,
    )

    for r, fragment in zip(results, os.listdir("./dataset/test/fragments")):
        source_image_name = image_test_data_csv[
            image_test_data_csv["Component"] == fragment
        ]["Original_image"].values[0]

        number_of_source_embeddings = len(
            database.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=f"{source_image_name}"),
                        ),
                    ]
                ),
                limit=1000,
            )[0]
        )
        
        top_k_sources = [rr.payload["source"] for rr in r]
        metrics = get_metrics(source_image_name, top_k_sources, number_of_source_embeddings)
        print(
            f"Fragment: {fragment}    Source: {source_image_name}.   Number of fragments in original: {number_of_source_embeddings}",
            f"\nTop k sources: {top_k_sources}\nMetrics: {metrics}\n\n\n",
        )


if __name__ == "__main__":
    # a = np.load("./data/embeddings.npy", mmap_mode="r+")
    # print(len(a))
    # # dimentionality_reduction()
    search_test()

    # database = QdrantDatabase(host="localhost", port=6333)
    # database.client.update_collection(
    #     collection_name=collection_name,
    #     hnsw_config=models.HnswConfigDiff(
    #                 m=32,  # Number of bi-directional links created for every new element during construction
    #                 ef_construct=300,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
    #                 full_scan_threshold=10000,
    #             ),
    # )