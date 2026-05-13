import os
import torch
import numpy as np
from nucleus.shared import NucleusQueues, Message, MessageType
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ["anger", "contempt", "disgust", "fear", "frustration",
          "gratitude", "joy", "love", "neutral", "sadness", "surprise"]

classifier_model = os.getenv("CLASSIFIER_MODEL")
if classifier_model is None:
    raise ValueError("CLASSIFIER_MODEL is not set")

classifier_threshold = float(os.getenv("CLASSIFIER_THRESHOLD", "0.3"))

class Interpreter:
    def __init__(self, queues: NucleusQueues):
        self.queues = queues
        self.tokenizer = None
        self.model = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(classifier_model)
        self.model.eval()

    async def run(self):
        while True:
            message = await self.queues.ingenium_in.get()

            if message.type == MessageType.CHUNK_READY:
                chunks = message.payload["chunks"]
                texts = [c.text for c in chunks]
                turn_tags = self.classify(texts)

                await self.queues.kortex_assembly.put(
                    Message(
                        type=MessageType.TAGS_READY,
                        source="ingenium",
                        payload={"turn_tags": turn_tags},
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
            vector = {LABELS[i]: float(row[i]) for i in range(len(LABELS))}
            results.append(vector)
        
        return results