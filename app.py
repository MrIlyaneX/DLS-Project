from re import I
from code.NomicEmbedder import NomicEmbedder
from code.QdrantDatabase import QdrantDatabase

collection_name: str = "image_vector_store_v1"
emb_size: int = 768
image_directory: str = ".dataset/"


def initialize() -> tuple[NomicEmbedder, QdrantDatabase]:
    embedder = NomicEmbedder(device="cpu", batch_size=4)
    database = QdrantDatabase(host="0.0.0.0", port=6333)

    slidig_window = ...
    cropper = ...

    database.create_collection(
        collection_name=collection_name,
        emb_size=emb_size,
        m=16,
        ef_construct=250,
        ef_search=100,
    )

    return embedder, database, [slidig_window, cropper]


def main():
    embedder, database, preprocessors = initialize()

    images = [preprocessor.preprocess() for preprocessor in preprocessors]

    image_names = images[0].keys()

    embeddings = [
        embedder.embed(*[images[i][image_name] for i in range(len(images))])
        for image_name in image_names
    ]

    database.add(
        vectors=embeddings,
        idx=image_names,
        payload=[{} for _ in range(len(embeddings))],
        collection_name=collection_name,
    )


if __name__ == "__main__":
    pass
