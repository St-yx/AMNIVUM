"""Opt-in Integrationstest gegen ein laufendes Qdrant (localhost:6333).

Überspringt sich automatisch, wenn Qdrant nicht erreichbar ist — normale `pytest`-Läufe
(CI, kein Docker) bleiben also unberührt. Deckt die echte qdrant-client-Query/Upsert-Grenze
ab, die die Unit-Test-Fakes NICHT abbilden — genau der Pfad, durch den der
`search`→`query_points`-Bruch geschlüpft ist.

Bewusst mit synthetischen Vektoren (keine Embedding-/Klassifikator-Modelle nötig → schnell)
und in einer pro Test frisch angelegten Wegwerf-Collection (keine Berührung mit memoria_*).
"""
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from nucleus.memoria import retriever as retr_mod
from nucleus.memoria.retriever import MemoriaRetriever
from nucleus.memoria.core import Chunk

HOST = os.getenv("VECTOR_DB_HOST", "localhost")
PORT = int(os.getenv("VECTOR_DB_PORT", "6333"))
DIM  = int(os.getenv("VECTOR_SIZE", "8"))


def _qdrant_reachable() -> bool:
    try:
        QdrantClient(host=HOST, port=PORT, timeout=1.0).get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant nicht erreichbar auf localhost:6333 — Integrationstest übersprungen",
)


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(DIM)


@pytest.fixture
def real_retriever():
    """MemoriaRetriever an echtem Qdrant, gegen frische Wegwerf-Collections."""
    client = QdrantClient(host=HOST, port=PORT)
    mid  = f"itest_mid_{uuid.uuid4().hex[:8]}"
    long = f"itest_long_{uuid.uuid4().hex[:8]}"
    for name in (mid, long):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

    # Funktionen im Retriever lesen diese Modul-Globals zur Laufzeit
    saved = (retr_mod.COLLECTION_MID, retr_mod.COLLECTION_LONG)
    retr_mod.COLLECTION_MID, retr_mod.COLLECTION_LONG = mid, long

    r = object.__new__(MemoriaRetriever)
    r.vecdb = client
    r._graph = {}
    r.graph_path = None
    try:
        yield r
    finally:
        retr_mod.COLLECTION_MID, retr_mod.COLLECTION_LONG = saved
        for name in (mid, long):
            client.delete_collection(name)


async def test_store_mid_then_retrieve_roundtrip(real_retriever):
    # 1. store_mid: echter Upsert + Novelty-query_points gegen leeres MID
    chunk = SimpleNamespace(
        text="Ich freue mich sehr über diesen wirklich schönen Tag heute.",
        embedding=_vec(1),
        topic_label="itest",
    )
    entry = SimpleNamespace(
        chunks=[chunk], turn_id="itest", knowledge_source="user1",
        turn_index=0, raw_tags=[{"joy": 0.9, "neutral": 0.1}],
    )
    await real_retriever.store_mid(entry, lambda c, tags, vecs: True)

    # 2. retrieve: echtes query_points (MID-Similarity) + scroll (Recency), kein LONG-Graph
    turn = [Chunk(text="Heute ist ein wirklich schöner, freudiger Tag.", embedding=_vec(2))]
    result = await real_retriever.retrieve(turn, current_turn_index=1)

    assert result.topics, "mindestens ein Topic erwartet"
    mid_sources = [c.source for c in result.mid_chunks]
    assert "MID" in mid_sources, "der zuvor geschriebene MID-Punkt muss retrievebar sein"


async def test_retrieve_on_empty_db_does_not_raise(real_retriever):
    # Leeres MID + leerer Graph: query_points/scroll dürfen nicht crashen (Regression
    # für search→query_points: hier starb vorher die MEMORIA-Task still).
    turn = [Chunk(text="Ein völlig neues Thema ohne jegliches Vorwissen hier.", embedding=_vec(3))]
    result = await real_retriever.retrieve(turn, current_turn_index=0)

    assert result.mid_chunks == []
    assert all(t.chunks == [] for t in result.topics)  # kein LONG-Treffer ohne Graph