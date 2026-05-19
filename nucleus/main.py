import asyncio
from nucleus.kortex import KortexInput
from nucleus.memoria import MemoriaCore
from nucleus.ingenium import Interpreter
from nucleus.shared import NucleusQueues, Services

async def main():
    services = Services()
    queues = NucleusQueues()
    await services.initialize()

    interpreter = Interpreter(queues, services)

    kortex = KortexInput(queues)
    memoria = MemoriaCore(queues, services)

    # start tasks - running parallel, waitig for msg
    asyncio.create_task(memoria.run())
    asyncio.create_task(interpreter.run())

    await kortex.receive("Hallo Violet", source="user")

    await asyncio.sleep(1)

asyncio.run(main())