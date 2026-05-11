import asyncio
from nucleus.shared import Message, MessageType

class KortexInput:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def receive(self, plaintext: str, source: str) -> str:
        turn_id = generate_turn_id()
        message = Message(
            type=MessageType.USER_INPUT,
            source=source,
            payload={"text": plaintext},
            turn_id=turn_id
        )

        await self.queues.memoria_in.put(message)

        return turn_id