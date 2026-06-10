import os
import json
import asyncio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

load_dotenv()

BUFFER_SIZE              = int(os.getenv("MEMORIA_BUFFER_SIZE", "20"))

# == Candidate pool ===================================================== #
LONG_CANDIDATES_PRIMARY  = int(os.getenv("MEMORIA_LONG_CANDIDATES_PRIMARY", "30"))
LONG_CANDIDATES_SIDE     = int(os.getenv("MEMORIA_LONG_CANDIDATES_SIDE", "10"))
MID_CANDIDATES           = int(os.getenv("MEMORIA_MID_CANDIDATES", "30"))
MID_RECENT_TURNS         = int(os.getenv("MEMORIA_MID_RECENT_TURNS", "5"))

# == Topic detection ==================================================== #
TOPIC_DISTANCE_THRESHOLD = float(os.getenv("MEMORIA_TOPIC_DISTANCE_THRESHOLD", "0.40"))
TOPIC_MAX                = int(os.getenv("MEMORIA_TOPIC_MAX", "3"))

# == Source guarantees ================================================== #
GUARANTEE_USER0          = int(os.getenv("MEMORIA_GUARANTEE_USER0", "3"))   # AI
GUARANTEE_USER1          = int(os.getenv("MEMORIA_GUARANTEE_USER1", "8"))    # User
GUARANTEE_WORLD          = int(os.getenv("MEMORIA_GUARANTEE_WORLD", "2"))   # World


@dataclass
class RetrievedChunk:
    vecdb_id:           str
    text:               str
    source:             str         # "LONG" | "MID"
    knowledge_source:   str         # "user0" | "user1" | "world" | "unknown"
    similarity:         float
    importance:         float
    cluster_id:         str | None  # None for MID
    tags:               dict        # LONG: clean_tags, MID: raw_tags
    conflict_candidate: bool = False


@dataclass
class TopicResult:
    topic_vec:  np.ndarray              # Weighted avg vector of topic group
    cluster_id: str | None              # None = unknown topic, no LONG entry
    label:      str | None              # Word to describe topic for KORTEX
    chunks:     list[RetrievedChunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    topics:             list[TopicResult]       # 1–3 entrys, sorted by group size
    mid_chunks:         list[RetrievedChunk] = field(default_factory=list)
    turn_chunk_labels:  dict[int, str | None] = field(default_factory=dict)
 
    @property
    def has_knowledge(self) -> bool:
        return (
            any(t.chunks for t in self.topics) or
            bool(self.mid_chunks)
        )

# == Retriever ========================================================== #
class MemoriaRetriever:
    def __init__(self, vecdb: QdrantClient, graph_path:Path):
        self.vecdb     = vecdb
        self.graph_path = graph_path
        self._graph     = self._load_graph()

    def reload_graph(self):
        # reload graph after consolidator-update
        self._graph = self._load_graph()

    async def retrieve(
        self,
        turn_chunks: list[np.ndarray],
        current_turn_index: int,
    ) -> RetrievalResult:
        embeddings = [c.embedding for c in turn_chunks]
        texts      = [c.text      for c in turn_chunks]
 
        topic_data = self._extract_topics(embeddings, texts)

        # == LONG: query per topic ============================================== #
        topic_results = await asyncio.gather(
            *[self._retrieve_topic(vec, indices, turn_chunks, primary=(i == 0)) 
              for i, (vec, indices) in enumerate(topic_data)]
        )

        # turn_chunk_labels: index in turn_chunks → topic label
        # built from topic_data indices + resolved labels from topic_results
        turn_chunk_labels: dict[int, str | None] = {}
        for(_, indices), topic in zip(topic_data, topic_results):
            for idx in indices:
                turn_chunk_labels[idx] = topic.label


        # == MID: query per turn with larges topic vector as achor ============== #
        mid_chunks = await self._query_mid(topic_data[0][0], current_turn_index)
 
        return RetrievalResult(
            topics=list(topic_results),
            mid_chunks=mid_chunks,
            turn_chunk_labels=turn_chunk_labels,
        )

    
    # ======================================================================= #
    # Topic recognition                                                       #
    # ======================================================================= #

    def _extract_topics(
        self,
        embeddings: list[np.ndarray],
        texts:      list[str],
    ) -> list[np.ndarray]:

        # group chunks by Agglomerative Clustering (Cosine-Distance).
        # give one Weighted-Average-Vector per topic (max TOPIC_MAX).
        if len(embeddings) == 1:
            return [(embeddings[0], [0])]
 
        matrix = np.stack(embeddings)
 
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=TOPIC_DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(matrix)
 
        # group chunks by label
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)
 
        # largest groups first, only TOPIC_MAX
        sorted_groups = sorted(groups.values(), key=len, reverse=True)[:TOPIC_MAX]
 
        topic_vecs = []
        for indices in sorted_groups:
            group_embs  = matrix[indices]
            group_words = np.array([max(len(texts[i].split()), 1) for i in indices])
            weights     = group_words / group_words.sum()
            vec         = np.average(group_embs, axis=0, weights=weights)
            topic_vecs.append((vec, list(indices)))
 
        return topic_vecs


    # =========================================================================== #
    # LONG-Retrieval per topic                                                    #
    # =========================================================================== #
    
    async def _retrieve_topic(self, 
        topic_vec: np.ndarray, 
        topic_indices: list[int],
        turn_chunks: list,
        primary:bool
    ) -> TopicResult:
        cluster_id = self._find_cluster(topic_vec)
        chunks: list[RetrievedChunk] = []
 
        # Label: try graph, then fallback
        label = None
        if cluster_id:
            label = self._graph[cluster_id].get("label")
            chunks = await self._query_long_cluster(topic_vec, cluster_id, primary)

        if not label:
            label = self._extract_label(turn_chunks, topic_indices)
 
        return TopicResult(
            topic_vec=topic_vec,
            cluster_id=cluster_id,  # None if no match
            label=label,
            chunks=chunks,          # empty if no match
        )
 
    def _find_cluster(self, topic_vec: np.ndarray) -> str | None:
        # similarity search against LONG-cluster-centroids, indexed in graph
        if not self._graph:
            return None
 
        best_id  = None
        best_sim = -1.0
 
        for cluster_id, data in self._graph.items():
            sim = self._cosine_similarity(topic_vec, np.array(data["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_id  = cluster_id
 
        return best_id
 
    async def _query_long_cluster(
        self,
        topic_vec:  np.ndarray,
        cluster_id: str,
        primary: bool,
    ) -> list[RetrievedChunk]:
        loop   = asyncio.get_event_loop()
        chunks: list[RetrievedChunk] = []
        limit = LONG_CANDIDATES_PRIMARY if primary else LONG_CANDIDATES_SIDE

        # == 1. Conflict Candidates (Must-Have, before guarantees) ============== #
        if primary:
            conflicts = await loop.run_in_executor(
                None,
                lambda: self.vecdb.scroll(
                    collection_name="LONG",
                    scroll_filter=Filter(must=[
                        FieldCondition(key="cluster_id",         match=MatchValue(value=cluster_id)),
                        FieldCondition(key="conflict_candidate", match=MatchValue(value=True)),
                    ]),
                    with_vectors=True,
                )[0]
            )
            for point in conflicts:
                chunks.append(self._point_to_chunk(point, "LONG", topic_vec))

        seen_ids = {c.vecdb_id for c in chunks}

        # == 2. Source guarantees parallel ====================================== #
        if primary:
            guaranteed = await self._query_long_with_guarantees(
                topic_vec, cluster_id, seen_ids, loop
            )
            chunks.extend(guaranteed)
            seen_ids |= {c.vecdb_id for c in guaranteed}

        # == 3. Rest of candidates by similarity ================================ #
        remaining = limit - len(chunks)
        if remaining > 0:
            sim_results = await loop.run_in_executor(
                None,
                lambda: self.vecdb.search(
                    collection_name="LONG",
                    query_vector=topic_vec.tolist(),
                    query_filter=Filter(must=[
                        FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id)),
                    ]),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            for point in sim_results:
                if str(point.id) not in seen_ids:
                    chunks.append(self._point_to_chunk(point, "LONG", topic_vec))
                    if len(chunks) >= limit:
                        break

        return chunks
    
    async def _query_long_with_guarantees(
        self,
        topic_vec:    np.ndarray,
        cluster_id:   str,
        exclude_ids:  set[str],
        loop,
    ) -> list[RetrievedChunk]:
        
        # Multiple parallel similarity queries - one for each knowledge source
        guarantees = [
            ("user0", GUARANTEE_USER0),     # AI
            ("user1", GUARANTEE_USER1),     # User
            ("world", GUARANTEE_WORLD),     # World
        ]

        async def query_source(knowledge_source: str, limit: int) -> list[RetrievedChunk]:
            results = await loop.run_in_executor(
                None,
                lambda: self.vecdb.search(
                    collection_name="LONG",
                    query_vector=topic_vec.tolist(),
                    query_filter=Filter(must=[
                        FieldCondition(key="cluster_id",      match=MatchValue(value=cluster_id)),
                        FieldCondition(key="knowledge_source",match=MatchValue(value=knowledge_source)),
                    ]),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            return [
                self._point_to_chunk(p, "LONG", topic_vec)
                for p in results
                if str(p.id) not in exclude_ids
            ]
        
        results = await asyncio.gather(
            *[query_source(src, lim) for src, lim in guarantees]
        )

        # put together, deduplicate
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for group in results:
            for chunk in group:
                if chunk.vecdb_id not in seen:
                    out.append(chunk)
                    seen.add(chunk.vecdb_id)
        
        return out
    

    # =========================================================================== #
    # MID-Retrieval                                                               #
    # =========================================================================== #

    async def _query_mid(
        self,
        topic_vec: np.ndarray,
        current_turn_index: int,
    ) -> list[RetrievedChunk]:
        
        # two parallel MID-Queries (theme and last n turns)

        loop = asyncio.get_event_loop()
        chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        # Similarity
        sim_results = await loop.run_in_executor(
            None,
            lambda: self.vecdb.search(
                collection_name="MID",
                query_vector=topic_vec.tolist(),
                limit=MID_CANDIDATES,
                with_payload=True,
                with_vectors=False,
            )
        )
        for point in sim_results:
            pid = str(point.id)
            if pid not in seen_ids:
                chunks.append(self._point_to_chunk(point, "MID", topic_vec))
                seen_ids.add(pid)

        # Recency: last n turns with metadata filter
        min_turn = max(0, current_turn_index - MID_RECENT_TURNS)
        recent_results = await loop.run_in_executor(
            None,
            lambda: self.vecdb.scroll(
                collection_name="MID",
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="turn_index",
                        range=Range(gte=min_turn),
                    ),
                ]),
                with_vectors=True,
            )[0]
        )
        for point in recent_results:
            pid = str(point.id)
            if pid not in seen_ids:
                chunks.append(self._point_to_chunk(point, "MID", topic_vec))
                seen_ids.add(pid)
 
        return chunks
    

    # =========================================================================== #
    # Helpers                                                                     #
    # =========================================================================== #
    
    def _point_to_chunk(self, point, source: str, ref_vec: np.ndarray) -> RetrievedChunk:
        payload = point.payload or {}
        vec = getattr(point, "vector", None)
        sim = self._cosine_similarity(ref_vec, np.array(vec)) if vec is not None else 0.0

        return RetrievedChunk(
            vecdb_id=           str(point.id),
            text=               payload.get("text", ""),
            source=             source,
            knowledge_source=   payload.get("knowledge_source", "unknown"),
            similarity=         getattr(point, "score", sim),
            importance=         payload.get("importance", 0.0),
            cluster_id=         payload.get("cluster_id"),
            tags=         payload.get("clean_tags") or payload.get("raw_tags") or {},
            conflict_candidate= payload.get("conflict_candidate", False)
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    
    def _load_graph(self) -> dict:
        if not self.graph_path.exists():
            return {}
        try:
            return json.loads(self.graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        
    def _extract_label(self, turn_chunks: list, indices: list[int]) -> str | None:
        if not indices or not turn_chunks:
            return None
        best = max(indices, key=lambda i: len(turn_chunks[i].text.split()))
        words = [w for w in turn_chunks[best].text.split() if len(w) > 3]
        return words[0].lower() if words else None