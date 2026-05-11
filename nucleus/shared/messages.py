from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    USER_INPUT = "user_input"
    LLM_INPUT = "llm_input"
    CHUNK_READY = "chunk_ready"
    TAGS_READY = "tags_ready"
    BUFFER_READY = "buffer_ready"
    AFFECT_READY = "affect_ready"

@dataclass
class Message:
    type: MessageType
    source: str     #"user", "llm", "kortex", ...
    payload: dict
    turn_id: str