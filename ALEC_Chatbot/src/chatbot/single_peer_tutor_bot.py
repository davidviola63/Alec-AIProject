# src/chatbot/single_peer_tutor_bot.py
from __future__ import annotations
import time, json

from src.chatbot.script.config import TOP_K, MAX_CONTEXT_TOKENS, RPM_LIMIT, TPM_LIMIT, RPD_LIMIT
from src.chatbot.script.debug import DEBUG_JUDGE, DEBUG_MODE, DEBUG_SAVE, dbg
from src.chatbot.script.rate_limiter import RateLimiter
from src.chatbot.script.gemini_client import load_gemini, gemini_json, gemini_text
from src.chatbot.script.retrieval import load_index_and_mapping, load_e5, search_rag
from src.chatbot.script.prompts import SYSTEM_CORE, JUDGE_INSTRUCTIONS, ANSWER_INSTRUCTIONS, build_context_block
from src.chatbot.script.single_conversation import Conversation
from src.chatbot.script.rf2_response_gate import decide_response
from src.chatbot.script.rf3_scaffolding import ScaffoldBundle, pick_cited_spans
from src.chatbot.script.prompts import SCAFFOLD_INSTRUCTIONS
from src.chatbot.script.gemini_client import gemini_scaffold_json
from src.chatbot.script import gemini_client as _gc

_gc.RATE = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)

def process_message(user_msg: str, index, mapping, emb_model, gen_model, conv: Conversation) -> str:
    # 1) Retrieval
    context = search_rag(index, mapping, emb_model, user_msg, top_k=TOP_K, max_ctx_tokens=MAX_CONTEXT_TOKENS)
    ctx_block = build_context_block(context)

    # 2) Judge
    judge_prompt = (
        f"SISTEMA:\n{SYSTEM_CORE}\n\nISTRUZIONI:\n{JUDGE_INSTRUCTIONS}\n\n"
        f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
        f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
        f"MESSAGGIO_UTENTE:\n{user_msg}\n"
    )
    try:
        judge = gemini_json(gen_model, judge_prompt)
        if DEBUG_JUDGE:
            dbg("\n=== GEMINI JUDGE RAW JSON ===")
            dbg(json.dumps(judge, ensure_ascii=False, indent=2))
            dbg("=== END JUDGE JSON ===\n")
        wrongness = float(judge.get("wrongness", 0.0))
    except Exception:
        wrongness = 0.0
        judge = {"wrongness": 0.0, "corrected": "", "citations": []}

    # 3) Decisione modalità
    decision = decide_response(wrongness)
    mode = decision.mode
    if DEBUG_MODE:
        dbg(f"[MODE] {mode.upper()} | wrongness={decision.wrongness:.2f}")

    # 4) Generazione
    base_answer = judge.get("corrected") or ""
    # 4a) Se mode = reinforce → nessuno scaffolding multi-livello: generazione testuale semplice
    if mode == "reinforce":
        answer_prompt = (
            f"SISTEMA:\n{SYSTEM_CORE}\n\n"
            f"Modalità: {mode}\n\n"
            f"ISTRUZIONI:\n{ANSWER_INSTRUCTIONS}\n\n"
            f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
            f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
            f"MESSAGGIO_UTENTE:\n{user_msg}\n\n"
            f"BOZZA_CORREZIONE (può essere vuota):\n{base_answer}\n"
        )
        text = gemini_text(gen_model, answer_prompt)

        if "Fonti" not in text:
            used = {(c['source'], c['chunk_id']) for c in context}
            fonti = "\n".join([f"- {s} ({cid})" for s, cid in used])
            text += f"\n\nFonti:\n{fonti}"
        return text

    # 4b) Se mode in {clarify, correct} → chiedi i 3 livelli in JSON in UNA chiamata
    scaffold_prompt = (
        f"SISTEMA:\n{SYSTEM_CORE}\n\n"
        f"Modalità: {mode}\n\n"
        f"ISTRUZIONI:\n{SCAFFOLD_INSTRUCTIONS}\n\n"
        f"STORIA (ultimi turni):\n{conv.history_block() or '(nessuna)'}\n\n"
        f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
        f"MESSAGGIO_UTENTE:\n{user_msg}\n\n"
        f"BOZZA_CORREZIONE (può essere vuota):\n{base_answer}\n"
    )
    sc = gemini_scaffold_json(gen_model, scaffold_prompt)

    # Normalizza e salva localmente i livelli 2-3 per /hint
    level1 = (sc.get("level1") or "").strip()
    level2 = (sc.get("level2") or "").strip()
    level3 = (sc.get("level3") or "").strip()
    # se il modello non ha messo "sources", agganciale dai citations del judge
    sources = sc.get("sources") or pick_cited_spans(judge.get("citations") or [], context, max_items=3)

    # salviamo i livelli rimanenti nella conversation
    conv.scaffold.set_bundle(ScaffoldBundle(level1=level1, level2=level2, level3=level3, sources=sources))

    # Mostra SOLO il primo livello ora
    text = level1
    if sources:
        text += "\n\nFonti:\n" + "\n".join(f"- {s}" for s in sources)

    return text

######################################################################################
######################################################################################
######################           MAIN                      ###########################
######################################################################################
######################################################################################

def main():
    print("PeerTutor RAG · Gemini 2.5 Flash")
    print("Comandi: /quit oppure /exit per uscire.")
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
        if user_msg.strip().lower() == "/hint":
            nxt = conv.scaffold.pop_next_hint()
            if nxt is None:
                print("\nTutor: Non ho altri indizi in sospeso. Prova con una nuova domanda.")
            else:
                srcs = conv.scaffold.sources_block()
                if srcs:
                    nxt = f"{nxt}\n\nFonti:\n{srcs}"
                print("\nTutor:", nxt)
                conv.add(user_msg, nxt)
            continue
        try:
            conv.scaffold.clear()
            reply = process_message(user_msg, index, mapping, emb_model, gen_model, conv)
        except RuntimeError as e:
            print(f"[RateLimit] {e}"); time.sleep(6); continue
        except Exception as e:
            print(f"[ERRORE] {e}"); continue
        print("\nTutor:", reply)
        conv.add(user_msg, reply)

if __name__ == "__main__":
    main()
