from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
from src.chatbot.script.config.variables import CLARIFY_MIN, CORRECT_MIN

Mode = Literal["reinforce", "clarify", "correct"]

@dataclass(frozen=True)
class ResponseDecision:
    mode: Mode
    wrongness: float


def decide_response(wrongness: Optional[float]) -> ResponseDecision:
    """Determina la modalità di risposta a partire dal livello di wrongness."""
    try:
        w = float(wrongness)
    except (TypeError, ValueError):
        w = 0.0
    w = max(0.0, min(1.0, w))

    if w >= CORRECT_MIN:
        return ResponseDecision(mode="correct", wrongness=w)
    elif w >= CLARIFY_MIN:
        return ResponseDecision(mode="clarify", wrongness=w)
    else:
        return ResponseDecision(mode="reinforce", wrongness=w)
