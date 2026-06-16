"""Tests für nucleus/shared/messages.py — der einfachste Einstieg (rein synchron)."""

from nucleus.shared.messages import (
    Message,
    MessageType,
    generate_turn_id,
)


def test_generate_turn_id_is_unique():
    ids = {generate_turn_id() for _ in range(1000)}
    assert len(ids) == 1000  # keine Kollisionen


def test_generate_turn_id_is_uuid_format():
    turn_id = generate_turn_id()
    # UUID4-String: 36 Zeichen, 4 Bindestriche
    assert isinstance(turn_id, str)
    assert len(turn_id) == 36
    assert turn_id.count("-") == 4


def test_message_holds_its_fields():
    msg = Message(
        type=MessageType.USER_INPUT,
        source="user",
        payload={"text": "hallo"},
        turn_id="t1",
    )
    assert msg.type is MessageType.USER_INPUT
    assert msg.source == "user"
    assert msg.payload == {"text": "hallo"}
    assert msg.turn_id == "t1"


def test_message_type_values_are_stable():
    # Schützt die String-Werte, auf die sich andere Module verlassen.
    assert MessageType.USER_INPUT.value == "user_input"
    assert MessageType.LLM_INPUT.value == "llm_input"
    assert MessageType.CHUNK_READY.value == "chunk_ready"
    assert MessageType.BUFFER_READY.value == "buffer_ready"
    assert MessageType.AFFECT_READY.value == "affect_ready"
