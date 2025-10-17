from __future__ import annotations
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from .config import HISTORY_MAX_TURNS
from .rf3_scaffolding import ScaffoldingStore

@dataclass
class ChatTurn:
    speaker: str
    message: str
    timestamp: float = time.time()

class ConversationMulti:
    """
    Gestisce una conversazione multi-utente con memoria degli ultimi turni.
    Ogni speaker mantiene un proprio scaffolding indipendente.
    """
    def __init__(self):
        self.turns: List[ChatTurn] = []
        self.scaffolds: Dict[str, ScaffoldingStore] = {}

    def add_turn(self, speaker: str, message: str):
        """Aggiunge un turno alla cronologia, ignorando i comandi /hint."""
        if message.strip().lower() == "/hint":
            return
        self.turns.append(ChatTurn(speaker, message))
        if len(self.turns) > HISTORY_MAX_TURNS:
            self.turns = self.turns[-HISTORY_MAX_TURNS:]

    def history_block(self, exclude: Optional[str] = None) -> str:
        """Restituisce gli ultimi turni (max HISTORY_MAX_TURNS) come testo."""
        h = []
        for t in self.turns[-HISTORY_MAX_TURNS:]:
            if exclude and t.speaker == exclude:
                continue
            h.append(f"{t.speaker}: {t.message}")
        return "\n\n".join(h)

    def scaffold_for(self, speaker: str) -> ScaffoldingStore:
        """Restituisce (o crea) lo scaffolding store per uno speaker."""
        if speaker not in self.scaffolds:
            self.scaffolds[speaker] = ScaffoldingStore()
        return self.scaffolds[speaker]
