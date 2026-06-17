"""Tests für die Affect-Mathematik in nucleus/ingenium/affect.py (Update 1).

`AffectUpdater` hängt an keinem Modell — nur NumPy. Es wird mit einem nicht
existierenden Pfad gebaut, sodass `_load()` auf den neutralen `DEFAULT_AFFECT`
zurückfällt (kein Schreiben, keine Datei nötig).

Kerninvariante: Update 1 ist eine *transiente* Stimmung — `global_affect` wird
NICHT verändert; nur `current_affect` trägt den Erinnerungs-Nudge.
"""

import numpy as np

from nucleus.ingenium.affect import AffectUpdater, LABELS, DEFAULT_AFFECT


def _vec(**kw):
    """Emotions-Vektor als dict; nicht genannte Labels sind 0.0."""
    return {label: float(kw.get(label, 0.0)) for label in LABELS}


def _updater(tmp_path):
    # Pfad existiert nicht -> _load() liefert DEFAULT_AFFECT
    return AffectUpdater(tmp_path / "missing_affect.json")


def _approx_eq(a: dict, b: dict) -> bool:
    return all(abs(a[label] - b[label]) < 1e-9 for label in LABELS)


# == _load ================================================================== #

def test_load_falls_back_to_default(tmp_path):
    au = _updater(tmp_path)
    assert au.state["global_affect"] == DEFAULT_AFFECT


# == Transiente Stimmung ==================================================== #

def test_update_1_does_not_mutate_baseline(tmp_path, make_retrieved_chunk, make_chunk):
    au = _updater(tmp_path)
    base = dict(au.state["global_affect"])

    buffer = [make_retrieved_chunk(knowledge_source="user0", topic_label="A",
                                   tags=_vec(joy=0.9))]
    out = au.update_1(
        buffer_chunks=buffer,
        turn_tags=[_vec(joy=0.8)],
        turn_chunks=[make_chunk("ich freue mich sehr", topic_label="A")],
        topic_labels={"topic1": "A", "topic2": None, "topic3": None},
    )

    # weder der interne Zustand noch das zurückgegebene global_affect ändern sich
    assert _approx_eq(au.state["global_affect"], base)
    assert _approx_eq(out["global_affect"], base)
    # current_affect weicht ab (Buffer nicht leer)
    assert not _approx_eq(out["current_affect"], base)


def test_update_1_payload_has_all_keys(tmp_path, make_retrieved_chunk, make_chunk):
    au = _updater(tmp_path)
    out = au.update_1(
        buffer_chunks=[make_retrieved_chunk(topic_label="A", tags=_vec(joy=0.5))],
        turn_tags=[_vec(joy=0.5)],
        turn_chunks=[make_chunk("a b c", topic_label="A")],
        topic_labels={"topic1": "A", "topic2": None, "topic3": None},
    )
    assert set(out) == {
        "global_affect", "current_affect", "topic_tags",
        "acceptance_tags", "drift", "flags",
    }


# == Gewichtung (Quelle + Topic-Rang) ====================================== #

def test_buffer_weighting_favours_higher_weight(tmp_path, make_retrieved_chunk, make_chunk):
    au = _updater(tmp_path)
    # gleiche Emotion-Stärke, aber user0+topic1 (Gewicht 1.0) vs user1+topic2 (0.25*0.5)
    buffer = [
        make_retrieved_chunk(vecdb_id="hi", knowledge_source="user0", topic_label="A",
                             tags=_vec(joy=1.0)),
        make_retrieved_chunk(vecdb_id="lo", knowledge_source="user1", topic_label="B",
                             tags=_vec(anger=1.0)),
    ]
    out = au.update_1(
        buffer_chunks=buffer,
        turn_tags=[_vec(joy=1.0)],
        turn_chunks=[make_chunk("x y", topic_label="A")],
        topic_labels={"topic1": "A", "topic2": "B", "topic3": None},
    )
    base = DEFAULT_AFFECT
    joy_push = out["current_affect"]["joy"] - base["joy"]
    anger_push = out["current_affect"]["anger"] - base["anger"]
    assert joy_push > anger_push > 0


# == topic_tags ============================================================= #

def test_topic_tags_group_by_label_and_skip_none(tmp_path, make_retrieved_chunk, make_chunk):
    au = _updater(tmp_path)
    turn_tags = [_vec(joy=0.4), _vec(sadness=0.6), _vec(anger=0.3)]
    turn_chunks = [
        make_chunk("a b c", topic_label="Haustiere"),
        make_chunk("d e f", topic_label="Familie"),
        make_chunk("g h i", topic_label=None),   # unbekanntes Topic -> nicht in topic_tags
    ]
    out = au.update_1(
        buffer_chunks=[],
        turn_tags=turn_tags,
        turn_chunks=turn_chunks,
        topic_labels={"topic1": "Haustiere", "topic2": "Familie", "topic3": None},
    )
    assert set(out["topic_tags"]) == {"Haustiere", "Familie"}


# == Drift / CONFLICT ======================================================= #

def test_high_drift_raises_single_conflict_flag(tmp_path, make_chunk):
    au = _updater(tmp_path)
    # reiner anger gegen den freundlichen DEFAULT-Grundton -> hoher Drift
    out = au.update_1(
        buffer_chunks=[],
        turn_tags=[_vec(anger=1.0)],
        turn_chunks=[make_chunk("wut pur jetzt", topic_label="Streit")],
        topic_labels={"topic1": "Streit", "topic2": None, "topic3": None},
    )
    assert out["drift"] > 0.45
    assert len(out["flags"]) == 1
    assert out["flags"][0] == {"type": "CONFLICT", "topic": "Streit"}


def test_consistent_turn_has_no_flag(tmp_path, make_chunk):
    au = _updater(tmp_path)
    # turn_tags nahe am Grundton -> niedriger Drift, kein Flag
    out = au.update_1(
        buffer_chunks=[],
        turn_tags=[_vec(**DEFAULT_AFFECT)],
        turn_chunks=[make_chunk("alles wie immer hier", topic_label="A")],
        topic_labels={"topic1": "A", "topic2": None, "topic3": None},
    )
    assert out["drift"] < 0.45
    assert out["flags"] == []


# == Edge: leerer Buffer ==================================================== #

def test_empty_buffer_keeps_baseline_as_mood(tmp_path, make_chunk):
    au = _updater(tmp_path)
    out = au.update_1(
        buffer_chunks=[],
        turn_tags=[_vec(joy=0.5)],
        turn_chunks=[make_chunk("a b c", topic_label="A")],
        topic_labels={"topic1": "A", "topic2": None, "topic3": None},
    )
    # kein Wissen -> kein Nudge -> current_affect == global_affect
    assert _approx_eq(out["current_affect"], DEFAULT_AFFECT)
