"""Tests für die Buffer-/Slot-Logik in nucleus/memoria/short.py.

Es wird mit kleinen, von Hand gebauten `RetrievalResult`-Objekten gearbeitet —
kein Retriever, keine Vektor-DB.
"""

import numpy as np

from nucleus.memoria.retriever import RetrievalResult, TopicResult
from nucleus.memoria.short import MemoriaShort


def _result(topic0_chunks, mid_chunks=None, side_chunks=None):
    """Baut ein minimales RetrievalResult.

    topic0 = Haupt-Topic. side_chunks landen als zweites Topic (Index 1).
    """
    topics = [TopicResult(topic_vec=np.zeros(8), cluster_id="c0", label="topic0",
                          chunks=topic0_chunks)]
    if side_chunks:
        topics.append(TopicResult(topic_vec=np.zeros(8), cluster_id="c1",
                                  label="topic1", chunks=side_chunks))
    return RetrievalResult(topics=topics, mid_chunks=mid_chunks or [])


# == update() =============================================================== #

def test_update_flags_insufficient_knowledge(make_retrieved_chunk):
    # Nur 2 LONG-Chunks (< SLOTS_TOPIC1 = 10) -> insufficient_knowledge.
    chunks = [
        make_retrieved_chunk(vecdb_id="w1", knowledge_source="world"),
        make_retrieved_chunk(vecdb_id="w2", knowledge_source="world"),
    ]
    short = MemoriaShort()
    selected = short.update(_result(chunks))

    assert short.insufficient_knowledge is True
    assert len(selected) == 2
    assert {c.vecdb_id for c in selected} == {"w1", "w2"}


def test_update_enough_knowledge_no_flag_and_dedups(make_retrieved_chunk):
    chunks = [
        make_retrieved_chunk(vecdb_id=f"u{i}", knowledge_source="user1",
                             importance=i / 100)
        for i in range(12)
    ]
    short = MemoriaShort()
    selected = short.update(_result(chunks))

    assert short.insufficient_knowledge is False
    ids = [c.vecdb_id for c in selected]
    assert len(ids) == len(set(ids))          # keine Duplikate
    assert len(selected) <= 20                 # BUFFER_SIZE


def test_update_conflict_candidate_is_always_included(make_retrieved_chunk):
    # Ein Konflikt-Kandidat mit niedriger Importance muss trotzdem rein.
    conflict = make_retrieved_chunk(vecdb_id="conf", knowledge_source="user1",
                                    importance=0.0, conflict_candidate=True)
    others = [
        make_retrieved_chunk(vecdb_id=f"u{i}", knowledge_source="user1",
                             importance=0.9)
        for i in range(5)
    ]
    short = MemoriaShort()
    selected = short.update(_result([conflict, *others]))

    assert "conf" in {c.vecdb_id for c in selected}


# == _pick_by_importance ==================================================== #

def test_pick_by_importance_sorts_and_limits(make_retrieved_chunk):
    chunks = [
        make_retrieved_chunk(vecdb_id="low", importance=0.1),
        make_retrieved_chunk(vecdb_id="high", importance=0.9),
        make_retrieved_chunk(vecdb_id="mid", importance=0.5),
    ]
    short = MemoriaShort()
    picked = short._pick_by_importance(chunks, limit=2)

    assert [c.vecdb_id for c in picked] == ["high", "mid"]


# == hold_turn_chunks / receive_raw_tags ==================================== #

def test_hold_and_receive_raw_tags_roundtrip(make_retrieved_chunk):
    short = MemoriaShort()
    held = [make_retrieved_chunk(vecdb_id="a")]
    short.hold_turn_chunks(held, turn_id="t1")

    returned = short.receive_raw_tags({"joy": 0.9}, turn_id="t1")
    assert returned is held

    # Nach dem Abholen ist das Pending geleert -> zweiter Aufruf gibt None.
    assert short.receive_raw_tags({"joy": 0.9}, turn_id="t1") is None


def test_receive_raw_tags_ignores_wrong_turn_id(make_retrieved_chunk):
    short = MemoriaShort()
    short.hold_turn_chunks([make_retrieved_chunk(vecdb_id="a")], turn_id="t1")

    assert short.receive_raw_tags({"joy": 0.9}, turn_id="other") is None
