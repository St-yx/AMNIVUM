import os
import asyncio
from pathlib import Path
from nucleus.kortex import KortexInput
from nucleus.memoria import MemoriaCore
from nucleus.ingenium import IngeniumCore
from nucleus.shared import NucleusQueues, Services

AFFECT_PATH = Path(os.getenv("AFFECT_PATH", "data/affect.json"))

async def main():
    services = Services()
    queues = NucleusQueues()
    await services.initialize()

    ingenium = IngeniumCore(queues, services, AFFECT_PATH)

    kortex = KortexInput(queues)
    memoria = MemoriaCore(queues, services)

    # start tasks - running parallel, waitig for msg
    asyncio.create_task(memoria.run())
    asyncio.create_task(ingenium.run())

    await kortex.receive("Hallo Violet", source="user")

    await asyncio.sleep(1)

asyncio.run(main())