import os
import torch
from torch.nn import Module
from nucleus.shared import Services
from transformers import AutoTokenizer


LABELS = ["anger", "contempt", "disgust", "fear", "frustration",
          "gratitude", "joy", "love", "neutral", "sadness", "surprise"]

classifier_model = os.getenv("CLASSIFIER_MODEL")
if classifier_model is None:
    raise ValueError("CLASSIFIER_MODEL is not set")

CLASSIFIER_THRESHOLD = float(os.getenv("INGENIUM_CLASSIFIER_THRESHOLD", "0.3"))


class Interpreter:
    # pure emotion classifier; the turn loop lives in IngeniumCore (core.py)
    def __init__(self, services: Services):
        self.tokenizer: AutoTokenizer = services.classifier_tokenizer
        self.model: Module = services.classifier_model

    @torch.no_grad()
    def classify(self, texts: list[str]) -> list[dict]:
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
