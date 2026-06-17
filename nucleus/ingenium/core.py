from pathlib import Path
from dataclasses import dataclass
from nucleus.ingenium.affect import AffectUpdater
from nucleus.ingenium.interpreter import Interpreter
from nucleus.shared import Message, MessageType, NucleusQueues, Services


@dataclass
class PendingTurn:
    # per-turn buffer: holds partial state until both messages arrived
    turn_tags:    list | None = None   # set after CHUNK_READY + classification
    buffer:       list | None = None   # set after BUFFER_READY
    turn_chunks:  list | None = None
    topic_labels: dict | None = None

    @property
    def ready(self) -> bool:
        return self.turn_tags is not None and self.buffer is not None


class IngeniumCore:
    # owns ingenium_in; classifies on CHUNK_READY, fires Update 1 on BUFFER_READY
    def __init__(self, queues: NucleusQueues, services: Services, affect_path: Path):
        self.queues = queues
        self.interpreter = Interpreter(services)
        self.affect = AffectUpdater(affect_path)
        self._pending: dict[str, PendingTurn] = {}

    async def run(self):
        while True:
            message = await self.queues.ingenium_in.get()
            turn = self._pending.setdefault(message.turn_id, PendingTurn())

            if message.type == MessageType.CHUNK_READY:
                chunks = message.payload["chunks"]
                texts = [c.text for c in chunks]
                turn.turn_tags = self.interpreter.classify(texts)
                await self.queues.memoria_in.put(
                    Message(
                        type=MessageType.TURN_TAGS_READY,
                        source="ingenium",
                        payload={"raw_tags": turn.turn_tags},
                        turn_id=message.turn_id,
                    )
                )

            elif message.type == MessageType.BUFFER_READY:
                turn.buffer       = message.payload["chunks"]
                turn.turn_chunks  = message.payload["turn_chunks"]
                turn.topic_labels = message.payload["topic_labels"]

            else:
                continue

            if turn.ready:
                await self._run_update_1(message.turn_id, turn)
                self._pending.pop(message.turn_id, None)

    async def _run_update_1(self, turn_id: str, turn: PendingTurn) -> None:
        payload = self.affect.update_1(
            buffer_chunks=turn.buffer,
            turn_tags=turn.turn_tags,
            turn_chunks=turn.turn_chunks,
            topic_labels=turn.topic_labels,
        )
        await self.queues.kortex_assembly.put(
            Message(
                type=MessageType.AFFECT_READY,
                source="ingenium",
                payload=payload,
                turn_id=turn_id,
            )
        )
