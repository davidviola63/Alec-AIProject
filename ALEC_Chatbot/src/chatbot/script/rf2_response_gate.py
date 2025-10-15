# src/chatbot/script/rf2_response_gate.py
# Python 3.11
"""
RF-2 Educational Policy
Il chatbot risponde sempre, adattando il tono e la profondità della risposta
in base al livello di scorrettezza (wrongness ∈ [0,1]).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

Mode = Literal["reinforce", "clarify", "correct"]

# soglie pedagogiche
CLARIFY_MIN = 0.15
CORRECT_MIN = 0.40


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
