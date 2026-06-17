"""Gemeinsame Test-Konfiguration und Fixtures.

WICHTIG: Die Env-Defaults am Anfang dieser Datei werden gesetzt, BEVOR irgendein
`nucleus`-Modul importiert wird. Das ist nötig, weil
`nucleus/shared/services.py` schon beim *Import* `int(os.getenv("VECTOR_DB_PORT"))`
und `int(os.getenv("VECTOR_SIZE"))` auswertet — ohne diese Werte würde bereits der
Import eines Test-Moduls mit `int(None)` fehlschlagen. `pytest` lädt `conftest.py`
vor allen Testdateien, daher ist hier der richtige Ort dafür.

`os.environ.setdefault` überschreibt eine echte lokale `.env` NICHT — es greift nur,
wenn die Variable noch nicht gesetzt ist. Dadurch laufen die Unit-Tests ohne `.env`
und ohne laufenden Qdrant/Docker.
"""

import os

# --- Env-Defaults: müssen vor jeglichem nucleus-Import gesetzt sein ---------- #
_ENV_DEFAULTS = {
    "VECTOR_DB_HOST": "localhost",
    "VECTOR_DB_PORT": "6333",
    "VECTOR_SIZE": "8",          # klein gehalten; echte Modelle werden nie geladen
    "EMBEDDING_MODEL": "test-embedding-model",
    "CLASSIFIER_MODEL": "test-classifier-model",
    "COLLECTION_LONG": "memoria_long",
    "COLLECTION_MID": "memoria_mid",
}
for _key, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

# Jetzt sind nucleus-Imports sicher.
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nucleus.memoria.core import Chunk  # noqa: E402
from nucleus.memoria.retriever import RetrievedChunk  # noqa: E402
from nucleus.shared.queues import NucleusQueues  # noqa: E402


@pytest.fixture
def queues():
    """Frische, leere Queues pro Test."""
    return NucleusQueues()


@pytest.fixture
def make_chunk():
    """Factory für `memoria.core.Chunk` mit handgesetztem Embedding.

    Aufruf z.B.: make_chunk("hallo welt", [1, 0, 0])
    Fehlt das Embedding, wird ein einfacher Default-Vektor verwendet.
    """
    def _make(text: str, embedding=None, topic_label=None) -> Chunk:
        if embedding is None:
            embedding = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return Chunk(
            text=text,
            embedding=np.asarray(embedding, dtype=float),
            topic_label=topic_label,
        )
    return _make


@pytest.fixture
def make_retrieved_chunk():
    """Factory für `retriever.RetrievedChunk` mit sinnvollen Defaults.

    Nur die Felder überschreiben, die der jeweilige Test braucht, z.B.:
        make_retrieved_chunk(vecdb_id="a", knowledge_source="world", importance=0.9)
    """
    def _make(
        vecdb_id: str = "id",
        text: str = "chunk text",
        source: str = "LONG",
        knowledge_source: str = "user1",
        similarity: float = 0.5,
        importance: float = 0.5,
        cluster_id=None,
        tags=None,
        conflict_candidate: bool = False,
        topic_label=None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            vecdb_id=vecdb_id,
            text=text,
            source=source,
            knowledge_source=knowledge_source,
            similarity=similarity,
            importance=importance,
            cluster_id=cluster_id,
            tags=tags if tags is not None else {},
            conflict_candidate=conflict_candidate,
            topic_label=topic_label,
        )
    return _make


class _FakeEmbedder:
    """Minimaler Ersatz für SentenceTransformer.

    `encode` liefert deterministische, normierte Vektoren basierend auf einem
    Hash des Textes — kein echtes Modell, keine Downloads. Reicht, damit
    `MemoriaCore.chunk()` durchläuft.
    """
    dim = 8

    def encode(self, texts):
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.standard_normal(self.dim))
        return np.asarray(vectors, dtype=float)


@pytest.fixture
def fake_embedder():
    return _FakeEmbedder()


class _FakeServices:
    """Ersatz für `Services` ohne geladene Modelle/DB.

    Wird `MemoriaCore` übergeben, das nur `services.embedder` und `services.vecdb`
    liest.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.vecdb = None


@pytest.fixture
def fake_services(fake_embedder):
    return _FakeServices(fake_embedder)
