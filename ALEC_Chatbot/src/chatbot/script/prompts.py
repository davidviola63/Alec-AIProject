from __future__ import annotations
from typing import List, Dict

SYSTEM_CORE = (
    "Sei un tutor di peer learning. Usa SOLO il contesto fornito (chunk top-k) per correggere o rispondere. "
    "Se una parte non è nel contesto, dillo esplicitamente. Cita le slide con (source, chunk_id). "
    "Stile: chiaro, sintetico."
)

JUDGE_INSTRUCTIONS = (
    "Compito: valuta quanto il messaggio utente è scorretto rispetto al CONTEXTO (0..1), "
    "dove 0 = tutto corretto, 1 = tutto errato. "
    "Restituisci sempre un JSON con le chiavi: wrongness (float 0..1), "
    "explanation (string), corrected (string, eventualmente vuota), citations (array di {source, chunk_id})."
)

ANSWER_INSTRUCTIONS = (
    "Compito: rispondi all'utente basandoti SOLO sul CONTEXTO. "
    "Modalità comportamentali:\n"
    "- reinforce: conferma ciò che è corretto, aggiungi 1 breve approfondimento utile.\n"
    "- clarify : conferma le parti corrette e correggi lievemente l'imprecisione (max 2 punti).\n"
    "- correct  : evidenzia chiaramente l'errore e fornisci una spiegazione sintetica con passaggi chiari.\n"
    "In ogni caso, chiudi con una sezione 'Fonti' con (source, chunk_id) usate."
)

def build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        lines.append(f"[CTX {i}] ({ch['source']} | {ch['doc_id']}:{ch['chunk_id']})\n{ch['text']}")
    return "\n\n---\n\n".join(lines)
