import os
import torch
import numpy as np
from torch.nn import Module
from nucleus.shared import NucleusQueues, Message, MessageType, Services
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ["anger", "contempt", "disgust", "fear", "frustration",
          "gratitude", "joy", "love", "neutral", "sadness", "surprise"]

classifier_model = os.getenv("CLASSIFIER_MODEL")
if classifier_model is None:
    raise ValueError("CLASSIFIER_MODEL is not set")

CLASSIFIER_THRESHOLD = float(os.getenv("INGENIUM_CLASSIFIER_THRESHOLD", "0.3"))

class Interpreter:
    def __init__(self, queues: NucleusQueues, services: Services):
        self.queues = queues
        self.tokenizer: AutoTokenizer = services.classifier_tokenizer
        self.model: Module = services.classifier_model

    async def run(self):
        while True:
            message = await self.queues.ingenium_in.get()

            if message.type != MessageType.CHUNK_READY:
                continue

            chunks = message.payload["chunks"]

            texts       = [c["text"]        for c in chunks]
            embeddings  = [c["embedding"]   for c in chunks]

            turn_tags = self._classify(texts)

            tagged_chunks = [
                {
                    "embedding": embeddings[i],
                    "turn_tags": turn_tags[i],
                }
                for i in range(len(chunks))
            ]

            await self.queues.ingenium_in.put(
                Message(
                    type=MessageType.TURN_TAGS_READY,
                    source="ingenium.interpreter",
                    payload={"tagged_chunks": tagged_chunks},
                    turn_id=message.turn_id
                )
            )

    @torch.no_grad()
    def _classify(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=192
        )
        probs = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()

        results = []
        for row in probs:
            vector = {
                LABELS[i]: float(row[i]) if row[i] >= CLASSIFIER_THRESHOLD else 0.0
                for i in range(len(LABELS))
                }
            results.append(vector)
        
        return results