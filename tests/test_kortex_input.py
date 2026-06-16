"""Async-Test für nucleus/kortex/input_handler.py.

Zeigt, wie man mit dem `queues`-Fixture und einer echten asyncio.Queue testet.
Dank asyncio_mode = "auto" reicht ein `async def` ohne @pytest.mark.asyncio.
"""

from nucleus.kortex.input_handler import KortexInput
from nucleus.shared.messages import MessageType


async def test_receive_user_input_puts_message_on_queue(queues):
    kortex = KortexInput(queues)

    turn_id = kortex.receive("Hallo Violet", source="user")
    turn_id = await turn_id  # receive ist async

    msg = queues.memoria_in.get_nowait()
    assert msg.type is MessageType.USER_INPUT
    assert msg.source == "user"
    assert msg.payload == {"text": "Hallo Violet"}
    assert msg.turn_id == turn_id


async def test_receive_llm_input_uses_llm_message_type(queues):
    kortex = KortexInput(queues)

    await kortex.receive("Violets Antwort", source="llm")

    msg = queues.memoria_in.get_nowait()
    assert msg.type is MessageType.LLM_INPUT
    assert msg.source == "llm"


async def test_receive_returns_unique_turn_ids(queues):
    kortex = KortexInput(queues)

    id1 = await kortex.receive("erste", source="user")
    id2 = await kortex.receive("zweite", source="user")

    assert id1 != id2
