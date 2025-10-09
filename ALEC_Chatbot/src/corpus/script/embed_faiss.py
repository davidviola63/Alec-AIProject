# src/corpus/script/embed_faiss.py
# Python 3.11

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Modello embedding consigliato per IT/multilingua (E5, ottimo per RAG)
MODEL_NAME = "intfloat/multilingual-e5-base"

# Path di progetto
PROJ_ROOT   = Path(__file__).resolve().parents[2]  # .../ALEC_Chatbot
DATA_DIR    = PROJ_ROOT / "corpus" / "data"
CORPUS_JSON = DATA_DIR / "corpus.json"
INDEX_DIR   = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH  = INDEX_DIR / "faiss_index.bin"
MAP_PATH    = INDEX_DIR / "mapping.jsonl"   # metadati in ordine vettori
CFG_PATH    = INDEX_DIR / "index_meta.json" # info su modello/dimensioni

def _load_corpus(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    # attesi campi: id, doc_id, chunk_id, source, text
    records = []
    for r in corpus:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        rid = str(r.get("id"))
        records.append({
            "id": rid,
            "doc_id": r.get("doc_id"),
            "chunk_id": r.get("chunk_id"),
            "source": r.get("source"),
            "text": text
        })
    if not records:
        raise RuntimeError("Corpus vuoto o senza campi 'text'.")
    return records

def _load_model(model_name: str):

    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model

def _encode_passages(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    E5 richiede i prefissi:
      - 'passage: ' per i documenti
      - 'query: '   per le query (vedi funzione di ricerca sotto)
    Usiamo normalizzazione L2 per FAISS (IndexFlatIP ~ cosine similarity).
    """
    prefixed = [f"passage: {t}" for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-ready
        show_progress_bar=True
    )
    return embs.astype("float32")

def build_index():
    # 1) Carica corpus
    records = _load_corpus(CORPUS_JSON)
    texts   = [r["text"] for r in records]

    # 2) Carica modello ed embeddizza
    print(f"[INFO] Carico modello: {MODEL_NAME}")
    model = _load_model(MODEL_NAME)
    embs  = _encode_passages(model, texts)
    dim   = embs.shape[1]

    # 3) Costruisci indice FAISS (cosine via inner product su vettori L2-normalizzati)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # 4) Salva indice + mapping + meta
    faiss.write_index(index, str(INDEX_PATH))

    with MAP_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            # scriviamo un JSON per riga nell'ordine degli embedding
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "model_name": MODEL_NAME,
        "dim": dim,
        "num_vectors": int(index.ntotal),
        "normalize": True,
        "similarity": "cosine_via_inner_product",
        "corpus_file": str(CORPUS_JSON)
    }
    with CFG_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Indice creato: {INDEX_PATH}")
    print(f"[OK] Mapping     : {MAP_PATH}")
    print(f"[OK] Meta        : {CFG_PATH}")
    print(f"[STATS] Vettori: {index.ntotal} | Dim: {dim}")

def _iter_mapping() -> List[Dict]:
    rows = []
    with MAP_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _encode_query(model, query: str) -> np.ndarray:
    # Per E5, prefisso 'query: '
    q = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)
    return q.astype("float32")

def search(query: str, top_k: int = 5):

    if not INDEX_PATH.exists() or not MAP_PATH.exists():
        raise SystemExit("Indice o mapping non trovati. Esegui prima --build.")

    print(f"[INFO] Carico indice: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))

    # opzionale: carica metadati (per controllo modello)
    if CFG_PATH.exists():
        meta = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        print(f"[INFO] Meta: {meta.get('model_name')} | dim={meta.get('dim')} | n={meta.get('num_vectors')}")

    print("[INFO] Carico modello (per embedding della query)…")
    model = _load_model(MODEL_NAME)
    q = _encode_query(model, query)

    # Ricerca
    scores, idxs = index.search(q, top_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    mapping = _iter_mapping()
    results = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0:  # FAISS restituisce -1 se meno vettori del top_k
            continue
        m = mapping[i]
        results.append({
            "rank": rank,
            "score": float(s),  # inner product con vettori normalizzati ~ cosine
            "id": m["id"],
            "doc_id": m["doc_id"],
            "chunk_id": m["chunk_id"],
            "source": m["source"],
            "text": m["text"]
        })

    print("\n=== RISULTATI ===")
    for r in results:
        print(f"[{r['rank']:02d}] score={r['score']:.4f}  {r['source']}  ({r['doc_id']}:{r['chunk_id']})")
        # Mostra un estratto del testo
        snippet = r["text"].replace("\n", " ")
        print(f"     {snippet}")
    return results

def parse_args():
    ap = argparse.ArgumentParser(description="Embedding di corpus.json e indice FAISS interrogabile")
    ap.add_argument("--build", action="store_true", help="Costruisce/aggiorna l'indice FAISS da corpus.json")
    ap.add_argument("--search", type=str, help="Esegue una query di test sull'indice")
    ap.add_argument("--top-k", type=int, default=5, help="Numero di risultati da mostrare (default: 5)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.build:
        build_index()
    elif args.search:
        search(args.search, top_k=args.top_k)
    else:
        print("Usi tipici:\n"
              "  python embed_faiss.py --build\n"
              "  python embed_faiss.py --search \"spiega A* e proprietà\" --top-k 8")
