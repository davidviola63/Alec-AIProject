from __future__ import annotations

import json
import uuid
from pathlib import Path

# === CONFIG ===
# Directory che contiene i file *.chunks.json prodotti dal chunking
INPUT_DIR = Path(r"C:\PycharmProjects\ALEC_Chatbot\src\corpus\data\chunked")

# Output: salviamo sempre in ../data rispetto a questo script
OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
OUTPUT_JSON = OUTPUT_DIR / "corpus.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"[ERRORE] Cartella input non trovata: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prendiamo tutti i file con estensione .json che finiscono in .chunks.json
    files = sorted([p for p in INPUT_DIR.glob("*.json") if p.is_file()])
    if not files:
        raise SystemExit("[ERRORE] Nessun file *.chunks.json trovato in {INPUT_DIR}")

    corpus: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (doc_id, chunk_id) per dedup

    for fp in files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[WARN] Salto {fp.name}: {e}")
            continue

        # Campi minimi attesi: doc_id, source, chunks:[{id, text}]
        doc_id = str(data.get("doc_id") or uuid.uuid4())
        source = str(data.get("source") or fp.name)

        chunks = data.get("chunks") or []
        if not isinstance(chunks, list):
            print(f"[WARN] Formato 'chunks' inatteso in {fp.name}; salto.")
            continue

        for ch in chunks:
            chunk_id = str(ch.get("id") or uuid.uuid4())
            text = (ch.get("text") or "").strip()

            if not text:
                continue  # salta chunk vuoti

            key = (doc_id, chunk_id)
            if key in seen:
                continue
            seen.add(key)

            # Record piatto, ideale per embedding: usa 'text' come input all'embedder
            record = {
                "id": f"{doc_id}:{chunk_id}",  # chiave univoca comoda per FAISS/sidecar
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": source,
                "text": text
            }
            corpus.append(record)

    # Salva come array JSON unico (richiesto)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"OK ✅  File uniti: {len(files)}")
    print(f"Chunk totali  : {len(corpus)}")
    print(f"Salvato in    : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
