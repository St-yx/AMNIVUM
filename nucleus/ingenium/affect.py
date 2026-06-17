import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# fixed label order, must match the classifier (interpreter.py)
LABELS = ["anger", "contempt", "disgust", "fear", "frustration",
          "gratitude", "joy", "love", "neutral", "sadness", "surprise"]

# == Source weighting (knowledge_source) ==================================== #
WEIGHT_AI    = float(os.getenv("AFFECT_WEIGHT_AI", "1.0"))    # user0
WEIGHT_WORLD = float(os.getenv("AFFECT_WEIGHT_WORLD", "0.5"))  # world
WEIGHT_USER  = float(os.getenv("AFFECT_WEIGHT_USER", "0.25"))  # user1

# == Topic weighting (topic rank) =========================================== #
WEIGHT_TOPIC1 = float(os.getenv("AFFECT_WEIGHT_TOPIC1", "1.0"))
WEIGHT_TOPIC2 = float(os.getenv("AFFECT_WEIGHT_TOPIC2", "0.5"))
WEIGHT_TOPIC3 = float(os.getenv("AFFECT_WEIGHT_TOPIC3", "0.25"))

# == Affect mixing ========================================================== #
LEARNING_RATE  = float(os.getenv("AFFECT_LEARNING_RATE", "0.05"))
ACCEPT_GLOBAL  = float(os.getenv("ACCEPT_WEIGHT_GLOBAL", "0.6"))
ACCEPT_TURN    = float(os.getenv("ACCEPT_WEIGHT_TURN", "0.4"))
DRIFT_HIGH     = float(os.getenv("DRIFT_HIGH", "0.45"))

SOURCE_WEIGHTS = {"user0": WEIGHT_AI, "world": WEIGHT_WORLD, "user1": WEIGHT_USER}

# neutral baseline used when no affect.json exists yet
DEFAULT_AFFECT = {
    "anger": 0.05, "contempt": 0.02, "disgust": 0.01, "fear": 0.03,
    "frustration": 0.08, "gratitude": 0.21, "joy": 0.35, "love": 0.18,
    "neutral": 0.42, "sadness": 0.06, "surprise": 0.09,
}


class AffectUpdater:
    def __init__(self, affect_path: Path):
        self.affect_path = affect_path
        self.state = self._load()

    # =========================================================================== #
    # State                                                                       #
    # =========================================================================== #

    def _load(self) -> dict:
        # read persisted global_affect, fall back to neutral default.
        # Update 1 only reads; persisting belongs to Update 2.
        if self.affect_path.exists():
            with open(self.affect_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "global_affect": dict(DEFAULT_AFFECT),
            "last_updated": datetime.now().isoformat(),
        }

    # =========================================================================== #
    # Update 1 — affect computation before the prompt (transient mood)            #
    # =========================================================================== #

    def update_1(
        self,
        buffer_chunks: list,    # RetrievedChunk: .tags, .knowledge_source, .topic_label
        turn_tags:     list[dict],   # one emotion vector per turn chunk
        turn_chunks:   list,    # Chunk: .text, .topic_label (index-aligned to turn_tags)
        topic_labels:  dict,    # {"topic1": label, "topic2": label|None, "topic3": ...}
    ) -> dict:
        global_vec = self._to_array(self.state["global_affect"])

        # --- Step 1: weighted average of retrieved memory tags ----------------- #
        buffer_avg = self._buffer_average(buffer_chunks, topic_labels)

        # --- Step 2: transient current_affect (global_affect stays untouched) --- #
        if buffer_avg is None:
            current_vec = global_vec.copy()
        else:
            current_vec = global_vec * (1 - LEARNING_RATE) + buffer_avg * LEARNING_RATE

        # --- Step 3: topic_tags (turn tags grouped by topic_label) ------------- #
        topic_tags = self._topic_tags(turn_tags, turn_chunks)

        # --- Step 4: drift of what was just said vs. the stable baseline ------- #
        turn_avg = self._turn_average(turn_tags, turn_chunks)
        if float(np.linalg.norm(turn_avg)) == 0.0:
            drift = 0.0
        else:
            drift = 1.0 - self._cosine(turn_avg, global_vec)

        # --- Step 5: acceptance_tags (expectation horizon for the response) ---- #
        acceptance_vec = global_vec * ACCEPT_GLOBAL + turn_avg * ACCEPT_TURN

        # --- Step 6: flags ----------------------------------------------------- #
        flags = []
        if drift > DRIFT_HIGH:
            topic = self._max_drift_topic(topic_tags, global_vec)
            flags.append({"type": "CONFLICT", "topic": topic})

        return {
            "global_affect":   self._to_dict(global_vec),    # stable baseline (unchanged)
            "current_affect":  self._to_dict(current_vec),   # transient mood for the prompt
            "topic_tags":      topic_tags,
            "acceptance_tags": self._to_dict(acceptance_vec),
            "drift":           float(drift),
            "flags":           flags,
        }

    # =========================================================================== #
    # Helpers                                                                     #
    # =========================================================================== #

    def _buffer_average(self, buffer_chunks: list, topic_labels: dict) -> np.ndarray | None:
        # source- and topic-weighted mean of buffer chunk tags
        topic_weight = self._topic_weight_map(topic_labels)

        total_w = 0.0
        acc = np.zeros(len(LABELS))
        for c in buffer_chunks:
            w = SOURCE_WEIGHTS.get(c.knowledge_source, WEIGHT_USER) \
                * topic_weight.get(c.topic_label, WEIGHT_TOPIC3)
            if w <= 0:
                continue
            acc += w * self._to_array(c.tags)
            total_w += w

        if total_w == 0.0:
            return None
        return acc / total_w

    def _topic_weight_map(self, topic_labels: dict) -> dict:
        # label → topic rank weight (1.0 / 0.5 / 0.25); None labels ignored
        ranks = (("topic1", WEIGHT_TOPIC1), ("topic2", WEIGHT_TOPIC2), ("topic3", WEIGHT_TOPIC3))
        mapping = {}
        for key, weight in ranks:
            label = topic_labels.get(key)
            if label is not None:
                mapping[label] = weight
        return mapping

    def _topic_tags(self, turn_tags: list[dict], turn_chunks: list) -> dict:
        # group turn tags by topic_label (skip None), word-count weighted mean
        groups: dict[str, list[tuple[float, np.ndarray]]] = {}
        for i in range(min(len(turn_tags), len(turn_chunks))):
            label = turn_chunks[i].topic_label
            if label is None:
                continue
            weight = max(len(turn_chunks[i].text.split()), 1)
            groups.setdefault(label, []).append((weight, self._to_array(turn_tags[i])))

        result = {}
        for label, items in groups.items():
            total_w = sum(w for w, _ in items)
            acc = np.zeros(len(LABELS))
            for w, vec in items:
                acc += w * vec
            result[label] = self._to_dict(acc / total_w)
        return result

    def _turn_average(self, turn_tags: list[dict], turn_chunks: list) -> np.ndarray:
        # word-count weighted mean over all turn tags
        n = min(len(turn_tags), len(turn_chunks))
        if n == 0:
            return np.zeros(len(LABELS))

        total_w = 0.0
        acc = np.zeros(len(LABELS))
        for i in range(n):
            weight = max(len(turn_chunks[i].text.split()), 1)
            acc += weight * self._to_array(turn_tags[i])
            total_w += weight

        if total_w == 0.0:
            return np.zeros(len(LABELS))
        return acc / total_w

    def _max_drift_topic(self, topic_tags: dict, global_vec: np.ndarray) -> str | None:
        # topic whose tags deviate most from the baseline
        best_label = None
        best_drift = -1.0
        for label, vec_dict in topic_tags.items():
            d = 1.0 - self._cosine(self._to_array(vec_dict), global_vec)
            if d > best_drift:
                best_drift = d
                best_label = label
        return best_label

    @staticmethod
    def _to_array(d: dict) -> np.ndarray:
        return np.array([float(d.get(label, 0.0)) for label in LABELS])

    @staticmethod
    def _to_dict(arr: np.ndarray) -> dict:
        return {label: float(arr[i]) for i, label in enumerate(LABELS)}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
