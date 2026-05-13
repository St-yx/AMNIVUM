import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from nucleus.memoria.retriever import RetrievalResult, RetrievedChunk

load_dotenv()

BUFFER_SIZE     = int(os.getenv("MEMORIA_BUFFER_SIZE", "20"))

MID_CAP         = int(os.getenv("MEMORIA_MID_CAP", "5"))   # max MID-Slots in Buffer
GUARANTEE_WORLD = int(os.getenv("MEMORIA_GUARANTEE_WORLD", "2"))   # World
GUARANTEE_USER0 = int(os.getenv("MEMORIA_GUARANTEE_MODEL", "3"))   # AI
GUARANTEE_USER1 = int(os.getenv("MEMORIA_GUARANTEE_USER", "5"))    # User

SLOTS_TOPIC1    = int(os.getenv("MEMORIA_SLOTS_TOPIC1", "10"))
SLOTS_SIDE      = int(os.getenv("MEMORIA_SLOTS_SIDE", "5"))


@dataclass
class SlotDef:
    key:        str
    cap:        int
    available:  int
    allocated:  int = 0

    @property
    def positive_diff(self) -> int:
        # pool has more chunks then available cap (soft)
        return max(0, self.available - self.allocated)
    
    @property
    def negative_diff(self) -> int:
        # cap has more space then available chunks (hard)
        return max(0, self.cap - self.allocated)

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
        self.insufficient_knowledge: bool = False
        self._pending: PendingTurnChunks | None = None


    # =========================================================================== #
    # Buffer update                                                               #
    # =========================================================================== #

    def update(self, result: RetrievalResult) -> list[RetrievedChunk]:
        side_chunks: list[RetrievedChunk] = []
        for i, topic in enumerate(result.topics):
            if i in (1, 2):
                side_chunks.extend(topic.chunks)

        side_available = len(side_chunks)
        side_allocated = min(side_available, SLOTS_SIDE)
        side_gap = SLOTS_SIDE - side_allocated
        
        slots       = self._compose_template(result, extra_pass2=side_gap, side_allocated=side_allocated)
        selected    = self._map_chunks(result, slots, side_chunks, side_allocated)

        long_allocated = sum(s.allocated for s in slots if s.key != "mid")
        self.insufficient_knowledge = long_allocated < SLOTS_TOPIC1
        
        self.buffer = selected
        return self.buffer
    

    # =========================================================================== #
    # Buffer Template                                                             #
    # =========================================================================== #

    def _compose_template(self, result: RetrievalResult, extra_pass2: int, side_allocated: int) -> list[SlotDef]:
        # passing through knowledge-slots in chain MID → World → AI → User
        # setting mask for slot sizes in working array

        t1_chunks  = result.topics[0].chunks if result.topics else []

        world_pool = sum(1 for c in t1_chunks if c.knowledge_source == "world")
        ai_pool    = sum(1 for c in t1_chunks if c.knowledge_source == "user0")
        user_pool  = sum(1 for c in t1_chunks if c.knowledge_source == "user1")
        mid_pool   = len(result.mid_chunks)

 
        slots: list[SlotDef] = [
            SlotDef("mid",   MID_CAP,         available=mid_pool),
            SlotDef("world", GUARANTEE_WORLD, available=world_pool),
            SlotDef("ai",    GUARANTEE_USER0, available=ai_pool),
            SlotDef("user",  GUARANTEE_USER1, available=user_pool),
        ]

    # == Pass 1 ================================================================= #
        for i, slot in enumerate(slots):
            slot.allocated = min(slot.available, slot.cap)
            diff = slot.negative_diff   # cap not reached

            if diff > 0 and i + 1 < len(slots):
                self._round_robin_increase(diff, slots[i + 1:])

    # == Pass 2 ================================================================= #
        core_budget = (BUFFER_SIZE - side_allocated) - sum(s.allocated for s in slots)
        remaining   = core_budget + extra_pass2
 
        soft_slots = [s for s in slots if s.positive_diff > 0]

        i = 0
        while remaining > 0 and soft_slots:
            s = soft_slots[i % len(soft_slots)]
            if s.positive_diff > 0:
                s.allocated += 1
                remaining -= 1
            else:
                soft_slots.remove(s) # depleted, leave round-robin
            i += 1

        return slots
    
    def _round_robin_increase(self, diff: int, targets:list[SlotDef]) -> None:
        # distributes diff as cap increase evenly across slots (soft)
        if not targets:
            return
        
        n = len(targets)
        base = diff // n
        remainder = diff % n

        for slot in targets:
            slot.cap += base
        
        for i in range(remainder):
            targets[i % n].cap += 1


    # =========================================================================== #
    # applying chunks to slots                                                    #
    # =========================================================================== #
    
    def _map_chunks(
        self,
        result: RetrievalResult,
        slots: list[SlotDef],
        side_chunks: list[RetrievedChunk],
        side_allocated: int,
    ) -> list[RetrievedChunk]:
        
        selected: list[RetrievedChunk]  = []
        seen_ids: set[str]              = set()

        t1_chunks = result.topics[0].chunks if result.topics else []
        by_source: dict[str, list[RetrievedChunk]] = {
            "world": [], "user0": [], "user1": []
        }
        for c in t1_chunks:
            ks = c.knowledge_source if c.knowledge_source in by_source else "user1"
            by_source[ks].append(c)
                
        slot_map = {s.key: s for s in slots}

        # == 1. Slots with conflicts ============================================ #
        for pool in by_source.values():
            for c in pool:
                if c.conflict_candidate and c.qdrant_id not in seen_ids:
                    selected.append(c)
                    seen_ids.add(c.qdrant_id)
 
        # == 2. AI (user0) ====================================================== #
        s = slot_map["ai"]
        already = sum(1 for c in selected if c.knowledge_source == "user0")
        remaining = s.allocated - already
        if remaining > 0:
            candidates = [c for c in by_source["user0"]
                          if c.qdrant_id not in seen_ids]
            picked = self._pick_by_importance(candidates, remaining)
            selected.extend(picked)
            seen_ids |= {c.qdrant_id for c in picked}
 
        # == World ============================================================== #
        s = slot_map["world"]
        candidates = [c for c in by_source["world"]
                      if c.qdrant_id not in seen_ids]
        picked = self._pick_by_importance(candidates, s.allocated)
        selected.extend(picked)
        seen_ids |= {c.qdrant_id for c in picked}
 
        # == User (user1) ======================================================= #
        s = slot_map["user"]
        candidates = [c for c in by_source["user1"]
                      if c.qdrant_id not in seen_ids]
        picked = self._pick_by_importance(candidates, s.allocated)
        selected.extend(picked)
        seen_ids |= {c.qdrant_id for c in picked}
 
        # == MID ================================================================ #
        s = slot_map["mid"]
        candidates = [c for c in result.mid_chunks if c.qdrant_id not in seen_ids]
        picked = self._pick_by_importance(candidates, s.allocated)
        selected.extend(picked)
 
        return selected
    
    def _pick_by_importance(
        self,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda c: c.importance, reverse=True)[:limit]


    # =========================================================================== #
    # Turn-Chunk-Holding                                                          #
    # =========================================================================== #

    def hold_turn_chunks(self, chunks: list, turn_id: str) -> None:
        self._pending = PendingTurnChunks(chunks=chunks, turn_id=turn_id)

    def receive_raw_tags(self, raw_tags:dict, turn_id: str) -> list | None:
        if self._pending is None or self._pending.turn_id != turn_id:
            return None
        
        self._pending.raw_tags = raw_tags
        chunks = self._pending.chunks
        self._pending = None
        return chunks