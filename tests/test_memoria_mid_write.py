"""Tests für store_mid() in MemoriaRetriever und den insufficient_knowledge-Payload.

Nutzt einen Fake-VecDB, der search/upsert aufzeichnet, ohne Qdrant.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from nucleus.memoria.retriever import MemoriaRetriever
from nucleus.memoria.short import MemoriaShort, PendingTurnChunks
from nucleus.memoria.core import MemoriaCore, Chunk


# =========================================================================== #
# Fake VecDB                                                                   #
# =========================================================================== #

class _FakeVecDB:
    def __init__(self, neighbors=None):
        self._neighbors = neighbors or []  # list of np.ndarray returned as "near vectors"
        self.upserted: list = []

    def search(self, collection_name, query_vector, limit, with_vectors, with_payload=True):
        # return fake neighbor points
        class _P:
            def __init__(self, vec):
                self.vector = vec.tolist()
                self.id = "fake"
                self.payload = {}
                self.score = 0.5
        return [_P(v) for v in self._neighbors]

    def upsert(self, collection_name, points):
        self.upserted.extend(points)


def _make_retriever(vecdb):
    r = object.__new__(MemoriaRetriever)
    r.vecdb = vecdb
    r._graph = {}
    r.graph_path = None
    return r


def _make_entry(chunks, raw_tags, knowledge_source="user1", turn_index=0):
    return PendingTurnChunks(
        chunks=chunks,
        turn_id="t1",
        knowledge_source=knowledge_source,
        turn_index=turn_index,
        raw_tags=raw_tags,
    )


def _gate_always_pass(chunk, raw_tags, cluster_vecs):
    return True


def _gate_always_fail(chunk, raw_tags, cluster_vecs):
    return False


def _chunk(text="hallo welt", dim=8, topic_label="test"):
    emb = np.ones(dim, dtype=float)
    return Chunk(text=text, embedding=emb, topic_label=topic_label)


# =========================================================================== #
# store_mid — Payload-Schema                                                   #
# =========================================================================== #

async def test_store_mid_writes_point_with_correct_schema():
    vecdb = _FakeVecDB()
    retriever = _make_retriever(vecdb)
    chunk = _chunk()
    raw_tags = [{"joy": 0.8, "neutral": 0.1}]
    entry = _make_entry([chunk], raw_tags, knowledge_source="user1", turn_index=5)

    await retriever.store_mid(entry, _gate_always_pass)

    assert len(vecdb.upserted) == 1
    p = vecdb.upserted[0]
    payload = p.payload
    assert payload["text"] == chunk.text
    assert payload["knowledge_source"] == "user1"
    assert payload["turn_index"] == 5
    assert payload["topic_label"] == "test"
    assert payload["cluster_id"] is None
    assert "raw_tags" in payload
    assert payload["importance"] == pytest.approx(0.8)  # max non-neutral


async def test_store_mid_skips_chunks_that_fail_gate():
    vecdb = _FakeVecDB()
    retriever = _make_retriever(vecdb)
    entry = _make_entry([_chunk()], [{"neutral": 1.0}])

    await retriever.store_mid(entry, _gate_always_fail)

    assert vecdb.upserted == []


async def test_store_mid_importance_from_max_non_neutral():
    vecdb = _FakeVecDB()
    retriever = _make_retriever(vecdb)
    raw_tags = [{"joy": 0.3, "sadness": 0.7, "neutral": 0.9}]
    entry = _make_entry([_chunk()], raw_tags)

    await retriever.store_mid(entry, _gate_always_pass)

    assert len(vecdb.upserted) == 1
    assert vecdb.upserted[0].payload["importance"] == pytest.approx(0.7)


async def test_store_mid_all_neutral_importance_is_zero():
    vecdb = _FakeVecDB()
    retriever = _make_retriever(vecdb)
    entry = _make_entry([_chunk()], [{"neutral": 0.9}])

    await retriever.store_mid(entry, _gate_always_pass)

    assert vecdb.upserted[0].payload["importance"] == pytest.approx(0.0)


# =========================================================================== #
# insufficient_knowledge in kortex_assembly payload                            #
# =========================================================================== #

async def test_buffer_ready_contains_insufficient_knowledge(queues, fake_services, tmp_path):
    from nucleus.memoria import session_log as sl_mod
    sl_mod.SESSION_LOG_DIR = tmp_path / "slog"

    core = object.__new__(MemoriaCore)
    core.queues = queues
    core.embedder = fake_services.embedder
    core.vectordb = None
    core.session_log = SimpleNamespace(append=lambda *a, **kw: None)
    core.turn_index = 0
    core._turn_vec_window = []

    # Minimal fake retriever: returns empty result (no LONG knowledge → insufficient)
    from nucleus.memoria.retriever import RetrievalResult, TopicResult
    empty_result = RetrievalResult(
        topics=[TopicResult(topic_vec=np.zeros(8), cluster_id=None, label="x", chunks=[])],
        mid_chunks=[],
        turn_chunk_labels={},
    )

    class _FakeRetriever:
        async def retrieve(self, chunks, turn_index):
            return empty_result

    core.retriever = _FakeRetriever()
    core.short = MemoriaShort()

    from nucleus.shared.messages import Message, MessageType
    await queues.memoria_in.put(Message(
        type=MessageType.USER_INPUT,
        source="kortex",
        payload={"text": "Hallo, wie geht es dir?"},
        turn_id="t1",
    ))

    task = asyncio.create_task(core.run())
    try:
        # Skip CHUNK_READY on ingenium_in, grab both BUFFER_READY
        _ = await asyncio.wait_for(queues.ingenium_in.get(), timeout=2.0)  # CHUNK_READY
        _ = await asyncio.wait_for(queues.ingenium_in.get(), timeout=2.0)  # BUFFER_READY
        kortex_msg = await asyncio.wait_for(queues.kortex_assembly.get(), timeout=2.0)
    finally:
        task.cancel()

    assert "insufficient_knowledge" in kortex_msg.payload
    assert kortex_msg.payload["insufficient_knowledge"] is True  # no LONG chunks
