from tqdm import tqdm
from code.NomicEmbedder import NomicEmbedder
from code.QdrantDatabase import QdrantDatabase
from code.WindowSlidingCut import SlidingWindowCut
from code.DetectionCut import DetectionCut

from PIL import Image

import pandas as pd

collection_name: str = "image_vector_store_v1"
emb_size: int = 768
k: int = 5
image_directory: str = "./dataset/train/"


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

    image_test_data_csv = pd.read_csv("test_images.csv")

    images = [Image.open("./dataset/test/fragments/" + img_path) for img_path in os.listdir("./dataset/test/fragments")]

    query_embeddings = embedder.embed(images)

    results = database.search(
        querry=query_embeddings,
        collection_name=collection_name,
        limit=k,
    )
    for r, fragment in zip(results, os.listdir("./dataset/test/fragments")):
        print(fragment, image_test_data_csv[image_test_data_csv["Component"] == fragment][["Original_image"]], "\nSources:")  
        for rr in r:
            print(rr.payload["source"])
        print("\n")

if __name__ == "__main__":
    search_test()
