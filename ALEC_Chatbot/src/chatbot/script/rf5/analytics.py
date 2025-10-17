# src/analytics.py
from dataclasses import dataclass, field
from collections import Counter
from typing import List, Dict
import time, math, re

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

_TAG_RE = re.compile(r'<[^>]+>')
def strip_tags(text: str) -> str:
    return _TAG_RE.sub(' ', text or '')

@dataclass
class TurnRecord:
    ts: float
    speaker: str
    text_len: int
    wrongness: float | None
    relevance: str | None     # "pertinente" | "non_pertinente"
    citations: List[str] = field(default_factory=list)

@dataclass
class UserStats:
    turns_total: int = 0
    turns_pertinent: int = 0
    turns_offtopic: int = 0
    hints_requested: int = 0
    wrongness_sum: float = 0.0
    q_scores: List[float] = field(default_factory=list)
    sources_used: Counter = field(default_factory=Counter)

    def add_turn(self, rec: TurnRecord):
        self.turns_total += 1

        if rec.relevance == "non_pertinente":
            self.turns_offtopic += 1
            return

        self.turns_pertinent += 1

        if rec.wrongness is not None:
            self.wrongness_sum += rec.wrongness
            q = clamp(1 - rec.wrongness)
            self.q_scores.append(q)

        for s in rec.citations:
            self.sources_used[s] += 1

    def add_hint(self):
        self.hints_requested += 1

    def summary(self) -> Dict:
        n = len(self.q_scores)
        q_mean = sum(self.q_scores) / n if n > 0 else 0
        avg_wrong = self.wrongness_sum / max(1, self.turns_pertinent)
        return {
            "turns_total": self.turns_total,
            "pertinenti": self.turns_pertinent,
            "offtopic": self.turns_offtopic,
            "hints": self.hints_requested,
            "avg_wrongness": round(avg_wrong, 2),
            "quality_mean": round(q_mean, 2),
            "top_sources": [n for n, _ in self.sources_used.most_common(3)],
        }

class Analytics:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.by_user: Dict[str, UserStats] = {}

    def add_user_turn(self, speaker: str, text: str, judge: dict):
        rec = TurnRecord(
            ts=time.time(),
            speaker=speaker,
            text_len=len(strip_tags(text).split()),
            wrongness=judge.get("wrongness"),
            relevance=judge.get("relevance", "pertinente"),
            citations=judge.get("citations", []),
        )
        self.by_user.setdefault(speaker, UserStats()).add_turn(rec)

    def add_hint(self, speaker: str):
        self.by_user.setdefault(speaker, UserStats()).add_hint()

    def get_stats_html(self, speaker: str) -> str:
        data = self.by_user.get(speaker)
        if not data:
            return "<p>Nessuna statistica disponibile.</p>"
        s = data.summary()
        srcs = ', '.join(s["top_sources"]) or "Nessuna"
        return (
            f"<p><b>Statistiche di partecipazione</b></p>"
            f"<p>Interventi totali: {s['turns_total']}<br>"
            f"Pertinenti: {s['pertinenti']}<br>"
            f"Fuori contesto: {s['offtopic']}<br>"
            f"Hint richiesti: {s['hints']}<br>"
            f"Correttezza interventi: {s['quality_mean']:.2f}<br>"
            f"Fonti usate: {srcs}</p>"
        )
