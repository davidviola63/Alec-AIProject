import json

from src.chatbot.script.config.debug import dbg, DEBUG_JUDGE, DEBUG_MODE
from src.chatbot.script.rf1.retrieval import search_rag, build_context_block
from src.chatbot.script.config.prompts import (
    SYSTEM_CORE, JUDGE_INSTRUCTIONS, ANSWER_INSTRUCTIONS, SCAFFOLD_INSTRUCTIONS
)
from src.chatbot.script.rf2.response_gate import decide_response
from src.chatbot.script.rf3.scaffolding import ScaffoldBundle, pick_cited_spans
from src.chatbot.script.core.gemini_client import gemini_json, gemini_text, gemini_scaffold_json
from src.chatbot.script.config.variables import TOP_K, MAX_CONTEXT_TOKENS
from src.chatbot.script.rf4.conversation_multi import ConversationMulti


def process_message(user_msg: str, index, mapping, emb_model, gen_model, conv: ConversationMulti, speaker: str = "User") -> str:
    # 1) Retrieval
    context = search_rag(index, mapping, emb_model, user_msg, top_k=TOP_K, max_ctx_tokens=MAX_CONTEXT_TOKENS)
    ctx_block = build_context_block(context)

    # 2) Judge
    judge_prompt = (
        f"SISTEMA:\n{SYSTEM_CORE}\n\n"
        f"ISTRUZIONI:\n{JUDGE_INSTRUCTIONS}\n\n"
        f"STORIA (ultimi turni, escluso {speaker}):\n{conv.history_block(exclude=speaker) or '(nessuna)'}\n\n"
        f"CONTESTO (chunk top-k):\n{ctx_block}\n\n"
        f"MESSAGGIO_UTENTE ({speaker}):\n{user_msg}\n"
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
    # === RF-5: tracciamento partecipazione e qualità ===
    # Recupera i nomi delle fonti dal retrieval RAG
    used_sources = list({chunk['source'] for chunk in context if 'source' in chunk})

    # Tracciamento completo RF-5
    if hasattr(conv, "analytics"):
        try:
            conv.analytics.add_user_turn(
                speaker=speaker,
                text=user_msg,
                judge=judge | {"citations": used_sources}  # unione: fonti del RAG
            )
        except Exception as e:
            dbg(f"[RF5] errore tracking analytics: {e}")
    # ====================================================
    base_answer = judge.get("corrected") or ""

    # 4a) reinforce → risposta diretta
    if mode == "reinforce":
        answer_prompt = (
            f"SISTEMA:\n{SYSTEM_CORE}\n\n"
            f"Modalità: {mode}\n\n"
            f"ISTRUZIONI:\n{ANSWER_INSTRUCTIONS}\n\n"
            f"STORIA:\n{conv.history_block(exclude=speaker) or '(nessuna)'}\n\n"
            f"CONTESTO:\n{ctx_block}\n\n"
            f"MESSAGGIO_UTENTE ({speaker}):\n{user_msg}\n\n"
            f"BOZZA_CORREZIONE:\n{base_answer}\n"
        )
        text = gemini_text(gen_model, answer_prompt)
        if "Fonti" not in text:
            used = {(c['source']) for c in context}
            fonti = "\n".join([f"# {s}" for s in used])
            text += f"\n\nFonti:\n{fonti}"
        return text

    # 4b) clarify/correct → scaffolding JSON
    scaffold_prompt = (
        f"SISTEMA:\n{SYSTEM_CORE}\n\n"
        f"Modalità: {mode}\n\n"
        f"ISTRUZIONI:\n{SCAFFOLD_INSTRUCTIONS}\n\n"
        f"STORIA:\n{conv.history_block(exclude=speaker) or '(nessuna)'}\n\n"
        f"CONTESTO:\n{ctx_block}\n\n"
        f"MESSAGGIO_UTENTE ({speaker}):\n{user_msg}\n\n"
        f"BOZZA_CORREZIONE:\n{base_answer}\n"
    )
    sc = gemini_scaffold_json(gen_model, scaffold_prompt)

    level1 = (sc.get("level1") or "").strip()
    level2 = (sc.get("level2") or "").strip()
    level3 = (sc.get("level3") or "").strip()
    sources = sc.get("sources") or pick_cited_spans(judge.get("citations") or [], context, max_items=3)

    conv.scaffold_for(speaker).set_bundle(
        ScaffoldBundle(level1=level1, level2=level2, level3=level3, sources=sources)
    )

    text = level1
    if sources:
        text += "\n\nFonti:\n" + "\n".join(f"- {s}" for s in sources)
    return text
