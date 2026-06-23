"""Tests für den INGENIUM-Koordinator in nucleus/ingenium/core.py.

`IngeniumCore` besitzt die `ingenium_in`-Schleife, puffert pro `turn_id` und
zündet Update 1, sobald turn_tags (CHUNK_READY) UND Buffer (BUFFER_READY) da sind.

Der echte `Interpreter` (lädt ein Modell) wird NICHT konstruiert: die Instanz wird
über `object.__new__` gebaut und mit einem Stub-Interpreter bestückt. Der echte
`AffectUpdater` (nur NumPy) läuft mit, gegen einen nicht existierenden Pfad.
"""

import asyncio
from types import SimpleNamespace

from nucleus.ingenium.core import IngeniumCore
from nucleus.ingenium.affect import AffectUpdater, LABELS
from nucleus.shared.messages import Message, MessageType


def _vec(**kw):
    return {label: float(kw.get(label, 0.0)) for label in LABELS}


def _make_core(queues, tmp_path, affect_path=None):
    # __init__ umgehen -> kein Modell-Load; Stub-Interpreter + echter AffectUpdater
    core = object.__new__(IngeniumCore)
    core.queues = queues
    core.affect = AffectUpdater(affect_path or (tmp_path / "missing_affect.json"))
    core._pending = {}
    core.interpreter = SimpleNamespace(
        classify=lambda texts: [_vec(joy=0.5) for _ in texts]
    )
    return core


def _chunk_msg(turn_id):
    return Message(
        type=MessageType.CHUNK_READY,
        source="memoria",
        payload={"chunks": [SimpleNamespace(text="hallo welt", topic_label=None)]},
        turn_id=turn_id,
    )


def _buffer_msg(turn_id, make_retrieved_chunk, source="user", topic_switch=False):
    return Message(
        type=MessageType.BUFFER_READY,
        source="memoria",
        payload={
            "chunks": [make_retrieved_chunk(knowledge_source="user0", topic_label="T",
                                            tags=_vec(joy=0.9))],
            "turn_chunks": [SimpleNamespace(text="hallo welt", topic_label="T")],
            "topic_labels": {"topic1": "T", "topic2": None, "topic3": None},
            "source": source,
            "topic_switch": topic_switch,
        },
        turn_id=turn_id,
    )


async def _drive(core, messages):
    for m in messages:
        await core.queues.ingenium_in.put(m)
    task = asyncio.create_task(core.run())
    try:
        out = await asyncio.wait_for(core.queues.kortex_assembly.get(), timeout=2.0)
    finally:
        task.cancel()
    return out


async def test_emits_affect_ready_normal_order(queues, tmp_path, make_retrieved_chunk):
    core = _make_core(queues, tmp_path)
    tid = "turn-1"
    out = await _drive(core, [_chunk_msg(tid), _buffer_msg(tid, make_retrieved_chunk)])

    assert out.type is MessageType.AFFECT_READY
    assert out.source == "ingenium"
    assert out.turn_id == tid
    assert set(out.payload) == {
        "global_affect", "current_affect", "topic_tags",
        "acceptance_tags", "drift", "flags",
    }
    assert core._pending == {}        # nach dem Feuern geleert


async def test_emits_affect_ready_reversed_order(queues, tmp_path, make_retrieved_chunk):
    # BUFFER_READY trifft vor CHUNK_READY ein -> pending puffert, feuert trotzdem genau einmal
    core = _make_core(queues, tmp_path)
    tid = "turn-2"
    out = await _drive(core, [_buffer_msg(tid, make_retrieved_chunk), _chunk_msg(tid)])

    assert out.type is MessageType.AFFECT_READY
    assert out.turn_id == tid
    assert core.queues.kortex_assembly.empty()   # kein zweites Paket
    assert core._pending == {}


async def test_chunk_ready_alone_does_not_fire(queues, tmp_path):
    # Ohne BUFFER_READY darf kein AFFECT_READY entstehen; pending bleibt offen.
    core = _make_core(queues, tmp_path)
    tid = "turn-3"
    await core.queues.ingenium_in.put(_chunk_msg(tid))
    task = asyncio.create_task(core.run())
    await asyncio.sleep(0.05)
    task.cancel()

    assert core.queues.kortex_assembly.empty()
    assert tid in core._pending
    assert core._pending[tid].turn_tags is not None
    assert core._pending[tid].buffer is None


async def test_update_2_fires_after_affect_ready(queues, tmp_path, make_retrieved_chunk):
    """Nach AFFECT_READY wird Update 2 non-blocking gefeuert; affect.json entsteht."""
    path = tmp_path / "affect.json"
    core = _make_core(queues, tmp_path, affect_path=path)
    tid  = "turn-u2"

    out = await _drive(core, [_chunk_msg(tid), _buffer_msg(tid, make_retrieved_chunk, source="llm")])
    assert out.type is MessageType.AFFECT_READY

    await asyncio.sleep(0.1)   # Update-2-Task ausführen lassen

    assert path.exists()       # _persist hat geschrieben


async def test_pending_eviction_at_max(queues, tmp_path):
    """Wird _MAX_PENDING überschritten, wird der älteste Eintrag evictet."""
    from nucleus.ingenium.core import _MAX_PENDING, PendingTurn

    core = _make_core(queues, tmp_path)
    for i in range(_MAX_PENDING):
        core._pending[f"old-{i}"] = PendingTurn()

    oldest_id = "old-0"
    assert oldest_id in core._pending

    await core.queues.ingenium_in.put(_chunk_msg("new-turn"))
    task = asyncio.create_task(core.run())
    await asyncio.sleep(0.05)
    task.cancel()

    assert len(core._pending) == _MAX_PENDING
    assert oldest_id not in core._pending
    assert "new-turn" in core._pending
