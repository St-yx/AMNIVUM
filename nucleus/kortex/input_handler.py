# amnivum/kortex/input_handler.py
import asyncio
from nucleus.shared.queues import NucleusQueues
from nucleus.shared import Message, MessageType, generate_turn_id


class KortexInput:
    def __init__(self, queues: NucleusQueues):
        self.queues = queues

    async def receive(self, plaintext: str, source: str) -> str:
        turn_id = generate_turn_id()
        msg_type = MessageType.LLM_INPUT if source == "llm" else MessageType.USER_INPUT
        message = Message(
            type=msg_type,
            source=source,
            payload={"text": plaintext},
            turn_id=turn_id
        )

        await self.queues.memoria_in.put(message)

        return turn_id