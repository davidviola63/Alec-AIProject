from __future__ import annotations
from typing import List, Tuple
from .config import HISTORY_MAX_TURNS

class Conversation:
    def __init__(self):
        self.turns: List[Tuple[str, str]] = []

    def add(self, user_msg: str, assistant_msg: str):
        self.turns.append((user_msg, assistant_msg))
        if len(self.turns) > HISTORY_MAX_TURNS:
            self.turns = self.turns[-HISTORY_MAX_TURNS:]

    def history_block(self) -> str:
        if not self.turns: return ""
        h = []
        for u, a in self.turns[-HISTORY_MAX_TURNS:]:
            h.append(f"Utente: {u}\nAlec: {a}")
        return "\n\n".join(h)
