from ast import Tuple
from typing import Any, List

import numpy
from torch import Tensor
import torch.nn.functional as F
from .Base.EmbedderBase import EmbedderBase
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class NomicEmbedder(EmbedderBase):
    vision_model: AutoModel
    processor: AutoImageProcessor
    batch_size: int
    device: str

    def __init__(self, device: str = "cpu", batch_size: int = 4, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )

        self.vision_model.to(device)
        self.device = device

        self.batch_size = batch_size

    def embed(self, images: List[Image.Image]) -> List:
        embeddings = []
        skipped = 0
        wrong_images = False
        for batch in range(0, len(images), self.batch_size):
            try:
                image_batch = self.processor(
                images[batch : min(len(images), batch + self.batch_size)], return_tensors="pt"
                )
                image_batch.to(self.device)
                img_emb_batch = self.vision_model(**image_batch).last_hidden_state
            
                embeddings_batch = F.normalize(img_emb_batch[:, 0], p=2, dim=1)
                embeddings.extend(embeddings_batch.numpy(force=True))
            except:
                skipped += self.batch_size
                #print([s.show() for s in images[batch : min(len(images), batch + self.batch_size)]])
                print("Batch skipped")
        return embeddings