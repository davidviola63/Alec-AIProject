from __future__ import annotations

import os
from pathlib import Path

# === PATH ===
PROJ_ROOT = Path(__file__).resolve().parents[4]  # .../ALEC_Chatbot
DATA_DIR  = PROJ_ROOT / "src" / "corpus" / "data"
INDEX_DIR = DATA_DIR / "index"
CORPUS_JSON = DATA_DIR / "corpus.json"
INDEX_PATH  = INDEX_DIR / "faiss_index.bin"
MAP_PATH    = INDEX_DIR / "mapping.jsonl"
CFG_PATH    = INDEX_DIR / "index_meta.json"
ENV_PATH    = PROJ_ROOT / ".env"

REPORT_DIR = PROJ_ROOT / "src" / "report"

# === MODELLI ===
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# === RAG ===
TOP_K = 5
MAX_CONTEXT_TOKENS = 1400
HISTORY_MAX_TURNS = 6

# === RATE LIMIT (free tier gemini)
RPM_LIMIT = 10
TPM_LIMIT = 250_000
RPD_LIMIT = 250
