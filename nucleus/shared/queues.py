import asyncio

class NucleusQueues:
    def __init__(self):
        self.kortex_in = asyncio.Queue()        # User/LLM > KORTEX
        self.memoria_in = asyncio.Queue()       # KORTEX > MEMORIA
        self.ingenium_in = asyncio.Queue()      # KORTEX > INGENIUM
        self.kortex_assembly = asyncio.Queue()  # MEMORIA + INGENIUM > KORTEX