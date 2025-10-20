from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScaffoldBundle:
    """Pacchetto di aiuti per un singolo turno utente."""
    level1: str
    level2: str
    level3: str
    sources: List[str] = field(default_factory=list)

    def as_list(self) -> List[str]:
        return [self.level1, self.level2, self.level3]

class ScaffoldingStore:
    """
     Gestisce i livelli di aiuto correnti di *un singolo utente*.
     Ogni speaker nella conversazione multiutente deve avere la propria istanza,
     gestita dal ConversationMulti.
    - set_bundle(...) salva (sovrascrive) l'ultimo pacchetto
    - pop_next_hint() restituisce level2 poi level3 e li consuma
    - clear() pulisce tutto
    """
    def __init__(self):
        self._pending: List[str] = []
        self._sources: List[str] = []

    def set_bundle(self, bundle: ScaffoldBundle):
        self._pending = bundle.as_list()
        # Mostreremo subito level1; lasciamo in pending level2 e level3
        if self._pending:
            self._pending = self._pending[1:]
        self._sources = bundle.sources or []

    def pop_next_hint(self) -> Optional[str]:
        if not self._pending:
            return None
        return self._pending.pop(0)

    def has_hints(self) -> bool:
        return bool(self._pending)

    def clear(self):
        self._pending.clear()
        self._sources.clear()

    def sources_block(self) -> str:
        if not self._sources:
            return ""
        return "\n".join(f"- {s}" for s in self._sources)

def pick_cited_spans(citations: list[dict], context: list[dict], max_items: int = 3) -> list[str]:
    """
    Riferimenti compatti (solo nomi delle fonti) per agganciare gli indizi al contesto.
    I chunk_id restano usati solo per il matching interno, non per la visualizzazione.
    """
    if not citations:
        return []

    # Mantieni la logica di matching basata su (source, chunk_id)
    wanted = {
        (c.get("source"), c.get("chunk_id"))
        for c in citations
        if c.get("source") and c.get("chunk_id")
    }

    seen_sources = set()
    out = []

    for ch in context:
        key = (ch.get("source"), ch.get("chunk_id"))
        src = ch.get("source")
        if key in wanted and src not in seen_sources:
            seen_sources.add(src)
            out.append(src)
            if len(out) >= max_items:
                break

    return out

