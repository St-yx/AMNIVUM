import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from nucleus.memoria.retriever import RetrievalResult, RetrievedChunk

load_dotenv()

BUFFER_SIZE     = int(os.getenv("MEMORIA_BUFFER_SIZE", "15"))
SLOTS_TOPIC1    = int(os.getenv("MEMORIA_SLOTS_TOPIC1", "10"))
SLOTS_TOPIC2    = int(os.getenv("MEMORIA_SLOTS_TOPIC2", "3"))
SLOTS_TOPIC3    = int(os.getenv("MEMORIA_SLOTS_TOPIC3", "2"))
MID_CAP         = int(os.getenv("MEMORIA_MID_CAP", "5"))   # max MID-Slots in Buffer


@dataclass
class PendingTurnChunks:
    # chunks wait for INGENIUM raw_tags
    chunks:     list
    turn_id:    str
    raw_tags:   dict | None = None

    @property
    def ready(self) -> bool:
        return self.raw_tags is not None
    

class MemoriaShort:
    def __init__(self):
        self.buffer: list[RetrievedChunk] = []
        self._pending: PendingTurnChunks | None = None


    # =========================================================================== #
    # Buffer update                                                               #
    # =========================================================================== #

    def update(self, result: RetrievalResult) -> list[RetrievedChunk]:
        selected:list[RetrievedChunk] = []

        # == Conflict-Slots - must have of primary topic ======================== #
        if result.topics:
            conflicts = [c for c in result.topics[0].chunks if c.conflict_candidate]
            selected.extend(conflicts)

        conflict_ids = {c.qdrant_id for c in selected}


        # == Topic-Slots ======================================================== #
        slot_map = [
            (0, SLOTS_TOPIC1),
            (1, SLOTS_TOPIC2),
            (2, SLOTS_TOPIC3),
        ]
        for topic_idx, slots in slot_map:
            if topic_idx >= len(result.topics):
                break
            candidates = [
                c for c in result.topics[topic_idx].chunks
                if c.qdrant_id not in conflict_ids
            ]
            picked = self._pick(candidates, slots)
            selected.extend(picked)
            conflict_ids |= {c.qdrant_id for c in picked}
        
        # == MID-Slots ========================================================== #
        selected = self._merge_mid(selected, result.mid_chunks)

        self.buffer = selected
        return self.buffer
    

    # =========================================================================== #
    # Turn-Chunk-Holding                                                          #
    # =========================================================================== #

    def hold_turn_chunks(self, chunks: list, turn_id: str) -> None:
        self._pending = PendingTurnChunks(chunks=chunks, turn_id=turn_id)

    def recieve_raw_tags(self, raw_tags:dict, turn_id: str) -> list | None:
        if self._pending is None or self._pending.turn_id != turn_id:
            return None
        
        self._pending.raw_tags = raw_tags
        chunks = self._pending.chunks
        self._pending = None
        return chunks

    # =========================================================================== #
    # Helpers                                                                     #
    # =========================================================================== #

    def _pick(self, chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        # best chunks by importance - conflicts already pre-sorted
        sorted_chunks = sorted(chunks, key=lambda c: c.importance, reverse=True)
        return sorted_chunks[:limit]

    def _merge_mid(
        self,
        selected: list[RetrievedChunk],
        mid_chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not mid_chunks:
            return selected

        # MID immer rein, niedrigste Importance-LONG raus wenn Buffer voll
        combined = selected + mid_chunks
        if len(combined) <= BUFFER_SIZE:
            return combined

        # Trenne LONG und MID
        long_chunks = [c for c in combined if c.source == "LONG"]
        mid_only    = [c for c in combined if c.source == "MID"]

        # LONG nach Importance sortieren, kürzen bis Platz für MID
        long_chunks.sort(key=lambda c: c.importance, reverse=True)
        long_chunks = long_chunks[:BUFFER_SIZE - len(mid_only)]

        return long_chunks + mid_only