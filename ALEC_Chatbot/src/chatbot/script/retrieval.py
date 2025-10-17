from __future__ import annotations
import json, math
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chatbot.script.config import INDEX_PATH, MAP_PATH, EMBEDDING_MODEL_NAME, TOP_K, MAX_CONTEXT_TOKENS
from .debug import DEBUG_CTX, dbg

def est_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # ~4 char/token

def load_index_and_mapping():
    import faiss
    if not INDEX_PATH.exists() or not MAP_PATH.exists():
        raise SystemExit("Indice o mapping non trovati. Esegui prima embed_faiss.py --build.")
    idx = faiss.read_index(str(INDEX_PATH))
    mapping: List[Dict] = []
    with MAP_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                mapping.append(json.loads(line))
    if not mapping:
        raise SystemExit("Mapping vuoto.")
    return idx, mapping

def load_e5():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

def embed_query(model, q: str) -> np.ndarray:
    qpref = f"query: {q}"
    v = model.encode([qpref], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

def search_rag(index, mapping, model, query: str, top_k: int = TOP_K, max_ctx_tokens: int = MAX_CONTEXT_TOKENS):
    qv = embed_query(model, query)
    scores, idxs = index.search(qv, top_k * 4)  # candidati extra
    rows = []
    for pos, i in enumerate(idxs[0]):
        if i < 0: continue
        m = mapping[i]
        rows.append({
            "score": float(scores[0][pos]),
            "id": m["id"], "doc_id": m["doc_id"], "chunk_id": m["chunk_id"],
            "source": m["source"], "text": m["text"]
        })
    rows.sort(key=lambda r: r["score"], reverse=True)

    context, used = [], 0
    for r in rows:
        t = est_tokens(r["text"])
        if used + t > max_ctx_tokens: continue
        context.append(r); used += t
        if len(context) >= top_k: break

    if DEBUG_CTX:
        dbg("[CTX] Chunks selezionati:")
        for r in context:
            dbg(f"  - {r['source']}::{r['doc_id']}:{r['chunk_id']} (score={r.get('score'):.4f})")
    return context
