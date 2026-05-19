import re
import os
import asyncio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from nucleus.memoria import MemoriaShort, MemoriaRetriever
from nucleus.shared import Message, MessageType, NucleusQueues, Services

load_dotenv()

MERGE_THRESHOLD = os.getenv("MEMORIA_MERGE_THRESHOLD", "0.75")
MAX_CHUNK_WORDS = os.getenv("MEMORIA_MAX_CHUNK_WORDS", "50")
MIN_CHUNK_WORDS = os.getenv("MEMORIA_MIN_CHUNK_WORDS", "5")
NOVELTY_THRESHOLD = os.getenv("MEMORIA_NOVELTY_THRESHOLD", "0.85")
TOPIC_SWITCH_THRESHOLD = float(os.getenv("MEMORIA_TOPIC_SWITCH_THRESHOLD", "0.65"))
TOPIC_WINDOW_SIZE      = int(os.getenv("MEMORIA_TOPIC_WINDOW_SIZE",        "5"))
GRAPH_PATH = Path(os.getenv("MEMORIA_GRAPH_PATH", "data/cluster_graph.json"))


@dataclass
class Chunk:
    text:       str
    embedding:  np.ndarray # final embedding for vectorDB

class MemoriaCore:
    def __init__(self, queues: NucleusQueues, services: Services):
        self.queues = queues
        self.embedder = services.embedder
        self.vectordb = services.vectordb
        self.retriever = MemoriaRetriever(self.vectordb, GRAPH_PATH)
        self.short = MemoriaShort()
        self.turn_index = 0
        self._turn_vec_window: list[np.ndarray] = []

    async def run(self):
        while True:
            message = await self.queues.memoria_in.get()

            if message.type in (MessageType.USER_INPUT, MessageType.LLM_INPUT):

                # 1. Chunking
                chunks = await self.chunk(message.payload["text"])
                if not chunks:
                    continue

                # 2. Calculate turn vector
                turn_vec = self._turn_vec(chunks)

                # 3. Check for topic change, update window
                topic_switch = self._check_topic_switch(turn_vec)
                self._update_window(turn_vec)

                # 4. Retrieval
                result = await self.retriever.retrieve(chunks, self.turn_index)

                # 5. Fill buffer
                selected = self.short.update(result)

                # 6. ´Hold turn chunks for MID-writing (with INGENIUM-Tags)
                self.short.hold_turn_chunks(chunks, message.turn_id)

                # 7. Chunks to INGENIUM for classification
                await self.queues.ingenium_in.put(
                    Message(
                        type=MessageType.CHUNK_READY,
                        source="memoria",
                        payload={"chunks": selected},
                        turn_id=message.turn_id
                    )
                )
                
                # 8. Buffer to KORTEX
                await self.queues.kortex_assembly.put(
                    Message(
                        type=MessageType.CHUNK_READY,
                        source="memoria",
                        payload={
                            "chunks": selected,
                            "topic_switch": topic_switch,
                        },
                        turn_id=message.turn_id
                    )
                )

                self.turn_index += 1

    async def embed(self, texts:list[str]) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embedder.encode,
            texts
        )

    async def chunk(self, text: str) -> list[str]:
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
                else:
                    # too short: append to next
                    next_text = current_text + " " + next_text
                    next_vec = (current_vec + next_vec) / 2
                
                current_text = next_text
                current_vec = next_vec

        # last chunk
        if len(current_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(current_text)
        elif chunks:
            # too short: append to last
            chunks[-1] = chunks[-1] + " " + current_text
        else:
            # Edge cases: chunks empty, nothing to append - let gate decide
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