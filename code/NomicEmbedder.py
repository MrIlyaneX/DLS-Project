from typing import Any, List

import torch.nn.functional as F

from PIL import Image
from Base.EmbedderBase import EmbedderBase
from transformers import AutoModel, AutoImageProcessor

class NomicEmbedder(EmbedderBase):
    vision_model: AutoModel
    processor: AutoImageProcessor
    batch_size: int

    def __init__(self, batch_size: int = 4, *args: Any, **kwargs: Any) -> None:
        super.__init__(*args, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        self.batch_size = batch_size

    def embed(self, images: List[Image.Image]) -> List:
        embeddings = []
        for batch in range(0, len(images), self.batch_size):
            print(images[batch : min(len(images), batch + self.batch_size)])
            image_batch = self.processor(
                images[batch : min(len(images), batch + self.batch_size)], return_tensors="pt"
            )
            img_emb_batch = self.vision_model(**image_batch).last_hidden_state
            embeddings_batch = F.normalize(img_emb_batch[:, 0], p=2, dim=1)
            embeddings.extend(embeddings_batch.detach().tolist())
        return embeddings