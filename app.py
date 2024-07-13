from re import I
from code.NomicEmbedder import NomicEmbedder
from code.QdrantDatabase import QdrantDatabase
from code.WindowSlidingCut import SlidingWindowCut
from code.DetectionCut import DetectionCut

collection_name: str = "image_vector_store_v1"
emb_size: int = 768
image_directory: str = "./dataset/train/"


def initialize() -> tuple[NomicEmbedder, QdrantDatabase]:
    embedder = NomicEmbedder(device="cpu", batch_size=4)
    database = QdrantDatabase(host="0.0.0.0", port=6333)

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


def main() -> None:
    embedder, database, preprocessors = initialize()

    images = [
        preprocessor.process_dataset(image_directory) for preprocessor in preprocessors
    ]

    image_names = images[0].keys()

    embeddings = [
        emb
        for image_name in image_names
        for emb in embedder.embed(
            [img for i in range(len(images)) for img in images[i][image_name]]
        )
    ]

    print(len(embeddings))

    database.add(
        vectors=embeddings,
        idx=[i for i in range(len(embeddings))],
        payload=[{"a": "0"} for _ in range(len(embeddings))],
        collection_name=collection_name,
    )


if __name__ == "__main__":
    main()
