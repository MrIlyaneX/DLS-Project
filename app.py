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
    )

    return embedder, database, [sliding_window, cropper]


def dimentionality_reduction():
    from sklearn.random_projection import GaussianRandomProjection

    embedder = NomicEmbedder(device="mps", batch_size=4)
    preprocessors = [SlidingWindowCut(size=300, step=270), DetectionCut()]

    images_count = len(preprocessors[0])

    for image_number in tqdm(range(images_count)):
        np.save(
            f"./data/embeddings/emb{image_number}",
            np.array(
                embedder.embed(
                    [
                        img
                        for i in range(len(preprocessors))
                        for img in preprocessors[i][image_number]["cropped_fragments"]
                    ]
                )
            ),
        )


def deload():
    cropper = DetectionCut()
    images_count = len(cropper)

    all_emeddings = []

    for image_number in tqdm(range(images_count)):
        all_emeddings.extend(
            np.load(f"./data/embeddings/emb{image_number}.npy").tolist()
        )

    np.save("./data/embeddings_all/all", np.array(all_emeddings))


def npy_to_db():
    database = QdrantDatabase(host="localhost", port=6333)

    cropper = DetectionCut()
    images_count = len(cropper)

    counter = 0

    for image_number in tqdm(range(images_count)):
        image_name = cropper[image_number]["image_name"]
        embeddings = np.load(f"./data/embeddings/emb{image_number}.npy").tolist()

        idx = range(counter, counter + len(embeddings))
        counter += len(embeddings)

        database.add(
            vectors=embeddings,
            idx=[i for i in idx],
            payload=[{"source": image_name} for _ in range(len(embeddings))],
            collection_name=collection_name,
        )


def process_dataset() -> None:
    embedder, database, preprocessors = initialize()

    images_count = len(preprocessors[0])
    counter = 0

    for image_number in tqdm(range(images_count)):
        image_name = preprocessors[1][image_number]["image_name"]
        image_embeddings = embedder.embed(
            [
                img
                for i in range(len(preprocessors))
                for img in preprocessors[i][image_number]["cropped_fragments"]
            ]
        )

        idx = range(counter, counter + len(image_embeddings))
        counter += len(image_embeddings)

        database.add(
            vectors=image_embeddings,
            idx=[i for i in idx],
            payload=[{"source": image_name} for _ in range(len(image_embeddings))],
            collection_name=collection_name,
        )


def search_test(collection_to_use):
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
        collection_name=collection_to_use,
        limit=k,
    )
    all_metrics = []
    for r, fragment in zip(results, os.listdir("./dataset/test/fragments")):
        source_image_name = image_test_data_csv[
            image_test_data_csv["Component"] == fragment
        ]["Original_image"].values[0]
        fragment_generation_method = image_test_data_csv[
            image_test_data_csv["Component"] == fragment
        ]["Method"].values[0]

        number_of_source_embeddings = len(
            database.client.scroll(
                collection_name=collection_to_use,
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
        metrics = get_metrics(
            source_image_name, top_k_sources, number_of_source_embeddings
        )
        # print(
        #     f"Fragment: {fragment}    Source: {source_image_name}.   Number of fragments in original: {number_of_source_embeddings}",
        #     f"\nTop k sources: {top_k_sources}\nMetrics: {metrics}\n\n\n",
        # )
        fragment_metrics = {
            "method": fragment_generation_method,  # one of the 4 strings each meaning the method ('window_train', 'window_validation', 'detection_train', 'detection_validation')
            "metrics": metrics,  # dictionary with metrics
        }

        all_metrics.append(fragment_metrics)

    return fragment_metrics


def run_tests_for_collections():
    import json

    names_original = ["image_vector_store_v1", "image_vector_store_v1_cosine"]

    path_results = "./results/"

    names_lower_dim = [
        "embeddings_small_pca_Eucl",
        "embeddings_small_gaussian_Eucl",
        "embeddings_small_AE_Eucl",
        "embeddings_small_pca_Cos",
        "embeddings_small_gaussian_Cos",
        "embeddings_small_AE_Cos",
    ]

    for col_name in names_original:
        results = search_test(collection_to_use=col_name)
        print(results)
        with open(path_results + col_name, "w") as outfile:
            json.dump(results, outfile)

    for col_name in names_lower_dim:
        results = search_test(collection_to_use=col_name)
        print(results)
        with open(path_results + col_name, "w") as outfile:
            json.dump(results, outfile)


def create_databases():
    from qdrant_client.models import Distance, VectorParams
    from qdrant_client import models

    col_name = "embeddings_small_pca"

    low_dataset = np.load(f"./data/{col_name}/pca.npy")

    cropper = DetectionCut()
    database = QdrantDatabase(host="localhost", port=6333)

    col_name += "_Cos"
    if not database.client.collection_exists(collection_name=col_name):
        database.client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=526,
                distance=Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=64,
                    ef_construct=526,
                ),
            ),
            hnsw_config=models.HnswConfigDiff(
                m=64,  # Number of bi-directional links created for every new element during construction
                ef_construct=526,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
                full_scan_threshold=10000,
            ),
        )

    images_count = len(cropper)
    counter = 0

    for image_number in tqdm(range(images_count)):
        image_name = cropper[image_number]["image_name"]

        embeddings = np.load(
            f"./data/embeddings/emb{image_number}.npy"
        ).tolist()  # find the #n of embeddings per image

        idx = range(counter, counter + len(embeddings))

        database.add(
            vectors=low_dataset[counter : counter + len(embeddings)],
            idx=[i for i in idx],
            payload=[{"source": image_name} for _ in range(len(embeddings))],
            collection_name=col_name,
        )

        counter += len(embeddings)

    database.client.update_collection(
        collection_name=col_name,
        hnsw_config=models.HnswConfigDiff(
            m=64,  # Number of bi-directional links created for every new element during construction
            ef_construct=256,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
            full_scan_threshold=10000,
        ),
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=5000),
    )


if __name__ == "__main__":
    run_tests_for_collections()
    # a = np.load("./data/embeddings.npy", mmap_mode="r+")
    # print(len(a))
    # # dimentionality_reduction()
    # process_dataset()

    # database = QdrantDatabase(host="localhost", port=6333)
    # database.client.update_collection(
    #     collection_name=collection_name,
    #     hnsw_config=models.HnswConfigDiff(
    #                 m=32,  # Number of bi-directional links created for every new element during construction
    #                 ef_construct=300,  # Size of the dynamic list for the nearest neighbors (used during the index construction)
    #                 full_scan_threshold=10000,
    #             ),
    # )
    # dimentionality_reduction()
    # deload()

    # create_databases()

# search_test()

# npy_to_db()
