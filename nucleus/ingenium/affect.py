from pathlib import Path

class AffectUpdater:
    def __init__(self, affect_path: Path):
        self.affect_path = affect_path
        self.state = self._load()

    def update_1(
        self,
        buffer_chunks: list,   # RetrievedChunk, mit topic_label + clean_tags
        turn_tags: list[dict], # pro Turn-Chunk ein Emotions-Vektor
    ) -> dict:
        # 1. cluster_affect pro Topic aus clean_tags berechnen
        # 2. acceptance_tags aus turn_tags + global_affect + cluster_affect
        # 3. gibt acceptance_tags + aktuellen affect zurück an KORTEX
        ...

'''
turn_tags liegen aus dem ersten Schritt im interpreter task (interpreter.py)
Datenstruktur und Pfad von affect.json zuerst festlegen (spezifikation in INGENIUM.txt)
'''