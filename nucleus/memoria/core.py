import re
import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass
from nucleus.shared import Message, MessageType, NucleusQueues, Services

load_dotenv()
mergeThreshold = os.getenv("MERGE_THRESHOLD")
maxChunkWords = os.getenv("MAX_CHUNK_WORDS")
minChunkWords = os.getenv("MIN_CHUNK_WORDS")

@dataclass
class Chunk:
    text:       str
    embedding:  np.ndarray # final embedding for vectorDB

class MemoriaCore:
    def __init__(self, queues: NucleusQueues, services: Services):
        self.queues = queues
        self.embedder = services.embedder
        self.vectordb = services.vectordb

    async def run(self):
        while True:
            message = await self.queues.memoria_in.get()

            if message.type in (MessageType.USER_INPUT, MessageType.LLM_INPUT):
                chunks = await self.chunk(message.payload["text"])

                await self.queues.kortex_assembly.put(
                    Message(
                        type=MessageType.CHUNK_READY,
                        source="memoria",
                        payload={"chunks": chunks},
                        turn_id=message.turn_id
                    )
                )
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

            if sim >= mergeThreshold and merged_word_count <= maxChunkWords:
                # combine: average vector
                current_text = merged_text
                current_vec = (current_vec + next_vec) / 2
            else:
                # end current chunk
                if len(current_text.split()) >= minChunkWords:
                    chunks.append(current_text)
                else:
                    # too short: append to next
                    next_text = current_text + " " + next_text
                    next_vec = (current_vec + next_vec) / 2
                
                current_text = next_text
                current_vec = next_vec

        # last chunk
        if len(current_text.split()) >= minChunkWords:
            chunks.append(current_text)
        elif chunks:
            # too short: append to last
            chunks[-1] = chunks[-1] + " " + current_text
        else:
            # Edge cases: chunks empty, nothing to append - let gate decide
            chunks.append(current_text)
        
        return chunks