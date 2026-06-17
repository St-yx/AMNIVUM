import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SESSION_LOG_DIR = Path(os.getenv("MEMORIA_SESSION_LOG_DIR", "data/session_log"))


class SessionLog:
    def __init__(self):
        SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self._path = SESSION_LOG_DIR / f"{ts}.txt"

    def append(self, turn_id: str, role: str, text: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        block = f"[{ts} | {role} | {turn_id}]\n{text}\n\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(block)
