# src/corpus/script/peer_tutor_bot.py
# Python 3.11
# Conversazione RAG + Gemini 2.5 Flash con correzione automatica >=30%

from __future__ import annotations
import os, time, json, math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# === CONFIG PATH ===
PROJ_ROOT = Path(__file__).resolve().parents[2]  # .../ALEC_Chatbot
DATA_DIR  = PROJ_ROOT / "src" / "corpus" / "data"
INDEX_DIR = DATA_DIR / "index"
CORPUS_JSON = DATA_DIR / "corpus.json"
INDEX_PATH  = INDEX_DIR / "faiss_index.bin"
MAP_PATH    = INDEX_DIR / "mapping.jsonl"
CFG_PATH    = INDEX_DIR / "index_meta.json"
ENV_PATH  = PROJ_ROOT / ".env"

# === EMBEDDING MODEL (E5) ===
MODEL_NAME = "intfloat/multilingual-e5-base"

# === RAG PARAMS ===
TOP_K = 5
MAX_CONTEXT_TOKENS = 1400   # budget contesto per chunk (stima token ≈ char/4)
OVERLAP_NOTE = "I contenuti provengono da slide del corso."

# === GEMINI ===
GEMINI_MODEL = "gemini-2.5-flash"
# limiti free tier dichiarati (soft-enforced via rate limiter locale)
RPM_LIMIT = 10          # richieste/min
TPM_LIMIT = 250_000     # token/min (stima)
RPD_LIMIT = 250         # richieste/giorno

# === CONVERSATION ===
HISTORY_MAX_TURNS = 6   # finestrella di contesto conversazionale
WRONGNESS_THRESHOLD = 0.30  # 30%

# === TOKEN STIMA ===
def est_tokens(text: str) -> int:
    # euristica ok per budgeting: ~ 4 char/token (italiano simile all’inglese)
    return max(1, math.ceil(len(text) / 4))

# === RATE LIMITER SEMPLICE ===
class RateLimiter:
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.win_min = []
        self.win_day = []
        self.token_min = 0
        self.curr_min = int(time.time() // 60)
        self.curr_day = int(time.time() // 86400)

    def update_windows(self):
        now_min = int(time.time() // 60)
        now_day = int(time.time() // 86400)
        if now_min != self.curr_min:
            self.curr_min = now_min
            self.win_min = []
            self.token_min = 0
        if now_day != self.curr_day:
            self.curr_day = now_day
            self.win_day = []

    def check(self, est_tokens_out: int = 1000):
        self.update_windows()
        if len(self.win_min) >= self.rpm:
            raise RuntimeError("Rate limit RPM locale raggiunto.")
        if self.token_min + est_tokens_out > self.tpm:
            raise RuntimeError("Rate limit TPM locale raggiunto.")
        if len(self.win_day) >= self.rpd:
            raise RuntimeError("Rate limit RPD locale raggiunto.")

    def commit(self, est_tokens_out: int = 1000):
        self.update_windows()
        self.win_min.append(time.time())
        self.win_day.append(time.time())
        self.token_min += est_tokens_out

RATE = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)

# === CARICAMENTO INDICE FAISS + MAPPING ===
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

# === EMBEDDING QUERY (E5) ===
def load_e5():

    return SentenceTransformer(MODEL_NAME, trust_remote_code=True)

def embed_query(model, q: str) -> np.ndarray:
    qpref = f"query: {q}"
    v = model.encode([qpref], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

# === RAG SEARCH ===
def search_rag(index, mapping, model, query: str, top_k: int = TOP_K, max_ctx_tokens: int = MAX_CONTEXT_TOKENS):
    qv = embed_query(model, query)
    scores, idxs = index.search(qv, top_k * 4)  # prendi più candidati, poi taglia su budget token
    idxs = [i for i in idxs[0].tolist() if i >= 0]
    cands = []
    for i in idxs:
        m = mapping[i]
        cands.append({
            "score": float(scores[0][idxs.index(i)]),
            "id": m["id"], "doc_id": m["doc_id"], "chunk_id": m["chunk_id"],
            "source": m["source"], "text": m["text"]
        })
    # Ordina per score decrescente
    cands.sort(key=lambda r: r["score"], reverse=True)
    # Pack nel budget di token
    context = []
    used = 0
    for r in cands:
        t = est_tokens(r["text"])
        if used + t > max_ctx_tokens:
            continue
        context.append(r)
        used += t
        if len(context) >= top_k:
            break
    return context

# === GEMINI CLIENT ===
def load_gemini():
    # Carica .env in modo esplicito e *deterministico*
    load_dotenv(dotenv_path=ENV_PATH, override=False)

    # Accetta sia GEMINI_API_KEY che GOOGLE_API_KEY
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            f"API key non trovata. Attese GEMINI_API_KEY o GOOGLE_API_KEY nel file {ENV_PATH} "
            "o tra le variabili d'ambiente del processo."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

# === PROMPTING ===
SYSTEM_CORE = (
    "Sei un tutor di peer learning. Usa SOLO il contesto fornito (chunk top-k) per correggere o rispondere. "
    "Se una parte non è nel contesto, dillo esplicitamente. Cita le slide con (source, chunk_id). "
    "Stile: chiaro, sintetico."
)

def build_context_block(chunks: List[Dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        lines.append(f"[CTX {i}] ({ch['source']} | {ch['doc_id']}:{ch['chunk_id']})\n{ch['text']}")
    return "\n\n---\n\n".join(lines)

# Fase 1: giudizio di correttezza
JUDGE_INSTRUCTIONS = (
    "Compito: valuta quanto il messaggio utente è scorretto rispetto al CONTEXTO (0..1), "
    "dove 0 = tutto corretto, 1 = tutto errato. Se >= 0.3, proponi una correzione basata SOLO sul contesto. "
    "Rispondi in JSON con le chiavi: wrongness (float 0..1), "
    "explanation (string), corrected (string, eventualmente vuota), citations (array di {source, chunk_id})."
)

# Fase 2: risposta correttiva/spiegazione
ANSWER_INSTRUCTIONS = (
    "Compito: rispondi all'utente basandoti SOLO sul CONTEXTO. "
    "Includi a fine risposta una sezione 'Fonti' con elenco (source, chunk_id) usate."
)

def gemini_json(model, prompt: str) -> dict:
    RATE.check(1000)
    resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    RATE.commit(1000)
    try:
        return json.loads(resp.text)
    except Exception:
        # fallback: prova a trovare il JSON nel testo
        t = resp.text
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end+1])
        raise

def gemini_text(model, prompt: str) -> str:
    RATE.check(2000)
    resp = model.generate_content(prompt)
    RATE.commit(2000)
    return resp.text.strip()

# === CONVERSATION STATE ===
class Conversation:
    def __init__(self):
        self.turns: List[Tuple[str, str]] = []  # (user, assistant)

    def add(self, user_msg: str, assistant_msg: str):
        self.turns.append((user_msg, assistant_msg))
        if len(self.turns) > HISTORY_MAX_TURNS:
            self.turns = self.turns[-HISTORY_MAX_TURNS:]

    def history_block(self) -> str:
        if not self.turns:
            return ""
        h = []
        for u, a in self.turns[-HISTORY_MAX_TURNS:]:
            h.append(f"Utente: {u}\nAlec: {a}")
        return "\n\n".join(h)

# === CORE PIPELINE ===
def process_message(user_msg: str, force_help: bool, index, mapping, emb_model, gen_model, conv: Conversation) -> str:
    # 1) Retrieval
    context = search_rag(index, mapping, emb_model, user_msg, top_k=TOP_K, max_ctx_tokens=MAX_CONTEXT_TOKENS)
    ctx_block = build_context_block(context)

    # 2) Valutazione correttezza (sempre, ma se /help è richiesto risponderemo comunque)
    judge_prompt = (
        f"SISTEMA:\n{SYSTEM_CORE}\n\nISTRUZIONI:\n{JUDGE_INSTRUCTIONS}\n\n"
        f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
        f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
        f"MESSAGGIO_UTENTE:\n{user_msg}\n"
    )
    try:
        judge = gemini_json(gen_model, judge_prompt)
        wrongness = float(judge.get("wrongness", 0.0))
    except Exception:
        # se fallisce il parsing, assumi neutro
        wrongness = 0.0
        judge = {"wrongness": 0.0, "explanation": "valutazione non disponibile", "corrected": "", "citations": []}

    should_correct = force_help or (wrongness >= WRONGNESS_THRESHOLD)

    # 3) Risposta (correttiva o esplicativa) basata SOLO sul contesto
    if should_correct:
        # prova a usare la correzione proposta, ma ricompone con istruzioni chiare
        citations = judge.get("citations") or []
        cite_lines = []
        for c in citations:
            src = c.get("source"); cid = c.get("chunk_id")
            if src and cid:
                cite_lines.append(f"- {src} ({cid})")
        cite_tail = "\n".join(cite_lines) if cite_lines else ""

        base_answer = judge.get("corrected") or ""

        answer_prompt = (
            f"SISTEMA:\n{SYSTEM_CORE}\n\nISTRUZIONI:\n{ANSWER_INSTRUCTIONS}\n\n"
            f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
            f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
            f"MESSAGGIO_UTENTE:\n{user_msg}\n\n"
            f"BOZZA_CORREZIONE (se vuota, ignora e genera tu):\n{base_answer}\n"
        )
        text = gemini_text(gen_model, answer_prompt)

        # assicurati di avere sezione Fonti
        if "Fonti" not in text:
            # aggiungi fonti dai chunk usati
            used = { (c['source'], c['chunk_id']) for c in context }
            fonti = "\n".join([f"- {s} ({cid})" for s,cid in used])
            text += f"\n\nFonti:\n{fonti}"
        return text
    else:
        # messaggio ok (<30% errato): rispondi aiutando/approfondendo su richiesta implicita
        # oppure conferma + chiarimento dal contesto
        answer_prompt = (
            f"SISTEMA:\n{SYSTEM_CORE}\n\nISTRUZIONI:\n{ANSWER_INSTRUCTIONS}\n"
            f"Se il messaggio è sostanzialmente corretto, conferma brevemente e aggiungi un chiarimento utile dal contesto.\n\n"
            f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
            f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
            f"MESSAGGIO_UTENTE:\n{user_msg}\n"
        )
        text = gemini_text(gen_model, answer_prompt)
        if "Fonti" not in text:
            used = { (c['source'], c['chunk_id']) for c in context }
            fonti = "\n".join([f"- {s} ({cid})" for s,cid in used])
            text += f"\n\nFonti:\n{fonti}"
        return text

# === CLI SEMPLICE ===
def main():
    print("PeerTutor RAG · Gemini 2.5 Flash")
    print("Comandi: /help per chiedere risoluzione esplicita; /quit per uscire.")
    index, mapping = load_index_and_mapping()
    emb_model = load_e5()
    gen_model = load_gemini()
    conv = Conversation()

    while True:
        try:
            user_msg = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCiao!")
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"/quit", "/exit"}:
            print("Ciao!")
            break

        force_help = user_msg.startswith("/help")
        q = user_msg[5:].strip() if force_help else user_msg

        try:
            reply = process_message(q, force_help, index, mapping, emb_model, gen_model, conv)
        except RuntimeError as e:
            print(f"[RateLimit] {e}")
            # attesa soft per rpm
            time.sleep(6)
            continue
        except Exception as e:
            print(f"[ERRORE] {e}")
            continue

        print("\nTutor:", reply)
        conv.add(user_msg, reply)

if __name__ == "__main__":
    main()
