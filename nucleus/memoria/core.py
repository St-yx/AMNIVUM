import re
import os
import asyncio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from nucleus.memoria.short import MemoriaShort
from nucleus.memoria.retriever import MemoriaRetriever
from nucleus.memoria.session_log import SessionLog
from nucleus.shared import Message, MessageType, NucleusQueues, Services

load_dotenv()

MERGE_THRESHOLD         = float(os.getenv("MEMORIA_MERGE_THRESHOLD", "0.75"))
MAX_CHUNK_WORDS         = int(os.getenv("MEMORIA_MAX_CHUNK_WORDS", "50"))
MIN_CHUNK_WORDS         = int(os.getenv("MEMORIA_MIN_CHUNK_WORDS", "5"))
NOVELTY_THRESHOLD       = float(os.getenv("MEMORIA_NOVELTY_THRESHOLD", "0.85"))
TOPIC_SWITCH_THRESHOLD  = float(os.getenv("MEMORIA_TOPIC_SWITCH_THRESHOLD", "0.65"))
TOPIC_WINDOW_SIZE       = int(os.getenv("MEMORIA_TOPIC_WINDOW_SIZE",        "5"))
GRAPH_PATH              = Path(os.getenv("MEMORIA_GRAPH_PATH", "data/cluster_graph.json"))


@dataclass
class Chunk:
    text:        str
    embedding:   np.ndarray # final embedding for vectorDB
    topic_label: str | None = None # set after retrieve(), used by KORTEX/INGENIUM

class MemoriaCore:
    def __init__(self, queues: NucleusQueues, services: Services):
        self.queues = queues
        self.embedder = services.embedder
        self.vectordb = services.vecdb
        self.retriever = MemoriaRetriever(self.vectordb, GRAPH_PATH)
        self.short = MemoriaShort()
        self.session_log = SessionLog()
        self.turn_index = 0
        self._turn_vec_window: list[np.ndarray] = []

    async def run(self):
        while True:
            message = await self.queues.memoria_in.get()

            if message.type in (MessageType.USER_INPUT, MessageType.LLM_INPUT):

                role = "user" if message.type == MessageType.USER_INPUT else "llm"
                self.session_log.append(message.turn_id, role, message.payload["text"])

                # 1. Chunking
                chunks = await self.chunk(message.payload["text"])
                if not chunks:
                    continue

                # 2. Calculate turn vector
                turn_vec = self._turn_vec(chunks)

                # 3. Check for topic change
                topic_switch = self._check_topic_switch(turn_vec)

                # 4. Update window
                self._update_window(turn_vec)

                # 5. First INGENIUM-Pass for turn classification
                await self.queues.ingenium_in.put(
                    Message(
                        type=MessageType.CHUNK_READY,
                        source="memoria",
                        payload={"chunks": chunks},   # raw turn chunks
                        turn_id=message.turn_id
                    )
                )
                
                # 6. Retrieval
                result = await self.retriever.retrieve(chunks, self.turn_index)

                # 7. Stamp topic labels onto turn chunks
                for i, chunk in enumerate(chunks):
                    chunk.topic_label = result.turn_chunk_labels.get(i)

                # 8. Fill buffer
                selected = self.short.update(result)

                # 9. Hold turn chunks for MID-writing (with INGENIUM-Tags from 5.)
                knowledge_source = (
                    "user1" if message.type == MessageType.USER_INPUT else "user0"
                )
                self.short.hold_turn_chunks(
                    chunks,
                    message.turn_id,
                    knowledge_source=knowledge_source,
                    turn_index=self.turn_index,
                )

                # 10. Buffer to INGENIUM for Affect Update 1
                #     topic_labels carries the rank→label mapping (topic1 = primary,
                #     result.topics is sorted by group size) so INGENIUM can weight
                #     by topic priority without re-deriving rank from labels.
                topic_labels = {
                    f"topic{i + 1}": (result.topics[i].label if i < len(result.topics) else None)
                    for i in range(3)
                }
                await self.queues.ingenium_in.put(
                    Message(
                        type=MessageType.BUFFER_READY,
                        source="memoria",
                        payload={"chunks":       selected, # selected memory chunks (LONG/MID) from buffer
                                 "turn_chunks":  chunks,    # raw turn chunks with topic_label
                                 "topic_labels": topic_labels,
                        },
                        turn_id=message.turn_id
                    )
                )
                
                # 11. Buffer to KORTEX
                await self.queues.kortex_assembly.put(
                    Message(
                        type=MessageType.BUFFER_READY,
                        source="memoria",
                        payload={
                            "chunks": selected,
                            "topic_switch": topic_switch,
                            "insufficient_knowledge": self.short.insufficient_knowledge,
                        },
                        turn_id=message.turn_id
                    )
                )

                self.turn_index += 1

            elif message.type == MessageType.TURN_TAGS_READY:
                raw_tags = message.payload["raw_tags"]
                entry = self.short.receive_raw_tags(raw_tags, message.turn_id)
                if entry is not None:
                    asyncio.create_task(
                        self._write_mid(entry),
                        name=f"mid-write-{message.turn_id}",
                    )

    async def _write_mid(self, entry) -> None:
        try:
            await self.retriever.store_mid(entry, self._importance_gate)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("MID-write failed for turn %s: %s", entry.turn_id, exc)

    async def embed(self, texts:list[str]) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embedder.encode,
            texts
        )

    async def chunk(self, text: str) -> list[Chunk]:
        sentences = self._split_sentences(text)

        if not sentences:
            return []
        
        if len(sentences) == 1:
            embedding = await self.embed([sentences[0]])
            return [Chunk(text=sentences[0], embedding=embedding[0])]
        
        merge_embeddings = await self.embed(sentences)
        texts = self._merge_sentences(sentences, merge_embeddings)

        final_embeddings = await self.embed(texts)
        return [Chunk(text=t, embedding=e) for t, e in zip(texts, final_embeddings)]
    
    def _split_sentences(self, text: str) -> list[str]:
        # cut at end, keep ending signs
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _merge_sentences(self, sentences: list[str], embeddings:np.ndarray) -> list[str]:
        chunks = []
        current_text = sentences[0]
        current_vec = embeddings[0].copy()

        for i in range(1, len(sentences)):
            next_text = sentences[i]
            next_vec = embeddings[i]

            sim = self._cosine_similarity(current_vec, next_vec)
            merged_text = current_text + " " + next_text
            merged_word_count = len(merged_text.split())

            if sim >= MERGE_THRESHOLD and merged_word_count <= MAX_CHUNK_WORDS:
                # combine: average vector
                current_text = merged_text
                current_vec = (current_vec + next_vec) / 2
            else:
                # end current chunk
                if len(current_text.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(current_text)
                elif merged_word_count <= MAX_CHUNK_WORDS:
                    # too short: append to next (still within MAX)
                    next_text = current_text + " " + next_text
                    next_vec = (current_vec + next_vec) / 2
                else:
                    # too short but merging would exceed MAX: emit as-is, let gate decide
                    chunks.append(current_text)

                current_text = next_text
                current_vec = next_vec

        # last chunk
        if len(current_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(current_text)
        elif chunks and len(chunks[-1].split()) + len(current_text.split()) <= MAX_CHUNK_WORDS:
            # too short: append to last (still within MAX)
            chunks[-1] = chunks[-1] + " " + current_text
        else:
            # too short but can't merge (no prior chunk or would exceed MAX) - let gate decide
            chunks.append(current_text)
        
        return chunks
    
    def _turn_vec(self, chunks: list[Chunk]) -> np.ndarray:
        # Weighted avg of all chunks by word count - represents Turn
        embeddings = np.stack([c.embedding for c in chunks])
        weights = np.array([max(len(c.text.split()), 1) for c in chunks])
        weights = weights / weights.sum()
        return np.average(embeddings, axis=0, weights=weights)
    
    def _check_topic_switch(self, turn_vec: np.ndarray) -> bool:
        # compares turn vector with sliding window avg of last N turns
        if not self._turn_vec_window:
            return False
        window_vec = np.mean(self._turn_vec_window, axis=0)
        sim = self._cosine_similarity(turn_vec, window_vec)
        return sim < TOPIC_SWITCH_THRESHOLD
    
    def _update_window(self, turn_vec: np.ndarray) -> None:
        # drop turn vector into Sliding Window
        self._turn_vec_window.append(turn_vec)
        if len(self._turn_vec_window) > TOPIC_WINDOW_SIZE:
            self._turn_vec_window.pop(0)

    def _importance_gate(
        self,
        chunk: Chunk,
        turn_tags: dict,
        cluster_vecs: list[np.ndarray],
    ) -> bool:
        # decides if chunks should be written to MID
        # too short
        if len(chunk.text.split()) < MIN_CHUNK_WORDS:
            return False
    
        # redundant
        if cluster_vecs:
            sims = [self._cosine_similarity(chunk.embedding, v) for v in cluster_vecs]
            if max(sims) > NOVELTY_THRESHOLD:
                return False
    
        # emotionally shallow - neutral without relevant emotions
        if turn_tags:
            neutral_score = turn_tags.get("neutral", 0.0)
            non_neutral = {k: v for k, v in turn_tags.items() if k != "neutral"}
            amplitude = max(non_neutral.values()) if non_neutral else 0.0
            if neutral_score > 0.7 and amplitude < 0.4:
                return False
    
        return True