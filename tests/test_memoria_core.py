"""Tests für die reine Logik in nucleus/memoria/core.py.

Hier wird kein echtes Embedding-Modell und keine Vektor-DB geladen: `MemoriaCore`
bekommt ein `fake_services` (Stub-Embedder, vecdb=None). Die getesteten Methoden
sind deterministisch — Embeddings werden für die meisten Tests von Hand vorgegeben.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from nucleus.memoria.core import MemoriaCore
from nucleus.memoria.retriever import RetrievalResult
from nucleus.shared.messages import Message, MessageType


@pytest.fixture
def core(queues, fake_services):
    """Eine MemoriaCore-Instanz mit gefälschten Services (keine Modelle/DB)."""
    return MemoriaCore(queues, fake_services)


# == _split_sentences ======================================================= #

def test_split_sentences_splits_on_punctuation(core):
    result = core._split_sentences("Hallo Welt. Wie geht es dir? Gut!")
    assert result == ["Hallo Welt.", "Wie geht es dir?", "Gut!"]


def test_split_sentences_empty_input(core):
    assert core._split_sentences("   ") == []


# == _cosine_similarity ===================================================== #

def test_cosine_similarity_identical_vectors(core):
    v = np.array([1.0, 2.0, 3.0])
    assert core._cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors(core):
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert core._cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero(core):
    # Division durch Null wird abgefangen.
    assert core._cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


# == _merge_sentences ======================================================= #

def test_merge_sentences_combines_similar(core):
    sentences = ["one two three four five", "six seven eight nine ten"]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])  # identisch -> sim = 1.0
    result = core._merge_sentences(sentences, embeddings)
    assert result == ["one two three four five six seven eight nine ten"]


def test_merge_sentences_keeps_dissimilar_apart(core):
    sentences = ["one two three four five", "six seven eight nine ten"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> sim = 0.0
    result = core._merge_sentences(sentences, embeddings)
    assert result == ["one two three four five", "six seven eight nine ten"]


def test_merge_sentences_appends_too_short_chunk_forward(core):
    # Erster Satz hat < MIN_CHUNK_WORDS (5) Wörter und wird nach vorne gemerged.
    sentences = ["a b", "c d e f g"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # nicht ähnlich
    result = core._merge_sentences(sentences, embeddings)
    assert result == ["a b c d e f g"]


# == _turn_vec ============================================================== #

def test_turn_vec_weights_by_word_count(core, make_chunk):
    chunks = [
        make_chunk("x", embedding=[1.0, 0.0]),            # 1 Wort
        make_chunk("p q r", embedding=[0.0, 1.0]),        # 3 Wörter
    ]
    result = core._turn_vec(chunks)
    # Gewichte 1:3 -> 0.25 * [1,0] + 0.75 * [0,1]
    assert result == pytest.approx([0.25, 0.75])


# == _check_topic_switch ==================================================== #

def test_check_topic_switch_empty_window_is_false(core):
    assert core._check_topic_switch(np.array([1.0, 0.0])) is False


def test_check_topic_switch_detects_change(core):
    core._turn_vec_window = [np.array([1.0, 0.0])]
    # orthogonaler Vektor -> sim 0 < TOPIC_SWITCH_THRESHOLD (0.65) -> Wechsel
    assert core._check_topic_switch(np.array([0.0, 1.0])) is True


def test_check_topic_switch_no_change_for_similar(core):
    core._turn_vec_window = [np.array([1.0, 0.0])]
    assert core._check_topic_switch(np.array([1.0, 0.0])) is False


# == _importance_gate ======================================================= #

def test_importance_gate_rejects_too_short(core, make_chunk):
    chunk = make_chunk("a b c", embedding=[1.0, 0.0])  # < 5 Wörter
    assert core._importance_gate(chunk, turn_tags={}, cluster_vecs=[]) is False


def test_importance_gate_rejects_redundant(core, make_chunk):
    chunk = make_chunk("one two three four five six", embedding=[1.0, 0.0])
    cluster_vecs = [np.array([1.0, 0.0])]  # identisch -> sim > NOVELTY_THRESHOLD
    assert core._importance_gate(chunk, turn_tags={}, cluster_vecs=cluster_vecs) is False


def test_importance_gate_rejects_emotionally_shallow(core, make_chunk):
    chunk = make_chunk("one two three four five six", embedding=[1.0, 0.0])
    turn_tags = {"neutral": 0.8, "joy": 0.2}  # neutral > 0.7, Amplitude < 0.4
    assert core._importance_gate(chunk, turn_tags=turn_tags, cluster_vecs=[]) is False


def test_importance_gate_accepts_meaningful_chunk(core, make_chunk):
    chunk = make_chunk("one two three four five six", embedding=[1.0, 0.0])
    turn_tags = {"neutral": 0.1, "joy": 0.9}  # starke Emotion
    assert core._importance_gate(chunk, turn_tags=turn_tags, cluster_vecs=[]) is True


# == chunk() — async-Beispiel =============================================== #

async def test_chunk_returns_chunks_for_single_sentence(core):
    # Dank asyncio_mode = "auto" braucht es hier KEIN @pytest.mark.asyncio.
    chunks = await core.chunk("This is a single test sentence here.")
    assert len(chunks) == 1
    assert chunks[0].text == "This is a single test sentence here."
    # Embedding-Dimension stammt aus dem Fake-Embedder (8).
    assert chunks[0].embedding.shape == (8,)


# == Plumbing: source + topic_switch in ingenium-BUFFER_READY =============== #

async def test_ingenium_buffer_ready_includes_source_and_topic_switch(core, queues):
    """ingenium-BUFFER_READY enthält 'source' und 'topic_switch' im Payload."""
    empty_result = RetrievalResult(topics=[], mid_chunks=[], turn_chunk_labels={})

    with patch.object(core.retriever, "retrieve", new=AsyncMock(return_value=empty_result)), \
         patch.object(core.session_log, "append"):
        await queues.memoria_in.put(Message(
            type=MessageType.USER_INPUT,
            source="kortex",
            payload={"text": "hallo welt wie geht es dir heute"},
            turn_id="t-src",
        ))
        task = asyncio.create_task(core.run())
        try:
            msg1 = await asyncio.wait_for(queues.ingenium_in.get(), timeout=2.0)
            msg2 = await asyncio.wait_for(queues.ingenium_in.get(), timeout=2.0)
        finally:
            task.cancel()
            await asyncio.sleep(0)   # Cancellation propagieren lassen

    msgs = {m.type: m for m in [msg1, msg2]}
    buf  = msgs.get(MessageType.BUFFER_READY)
    assert buf is not None, "Kein BUFFER_READY auf ingenium_in empfangen"
    assert "source"       in buf.payload
    assert "topic_switch" in buf.payload
    assert buf.payload["source"] == "user"
    assert isinstance(buf.payload["topic_switch"], bool)
