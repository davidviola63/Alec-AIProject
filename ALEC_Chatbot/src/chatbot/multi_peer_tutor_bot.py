
from __future__ import annotations
import time, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.chatbot.script.config import TOP_K, MAX_CONTEXT_TOKENS, RPM_LIMIT, TPM_LIMIT, RPD_LIMIT
from src.chatbot.script.debug import DEBUG_JUDGE, DEBUG_MODE, dbg
from src.chatbot.script.rate_limiter import RateLimiter
from src.chatbot.script.gemini_client import load_gemini, gemini_json, gemini_text, gemini_scaffold_json
from src.chatbot.script.retrieval import load_index_and_mapping, load_e5, search_rag
from src.chatbot.script.prompts import SYSTEM_CORE, JUDGE_INSTRUCTIONS, ANSWER_INSTRUCTIONS, SCAFFOLD_INSTRUCTIONS, build_context_block
from src.chatbot.script.rf2_response_gate import decide_response
from src.chatbot.script.rf3_scaffolding import ScaffoldBundle, pick_cited_spans
from src.chatbot.script.conversation_multi import ConversationMulti

# Inizializzazione globale
_gc_rate = RateLimiter(RPM_LIMIT, TPM_LIMIT, RPD_LIMIT)

index, mapping = load_index_and_mapping()
print("DEBUG → Tipo di index:", type(index))
print("DEBUG → Modulo di provenienza:", getattr(index, "__module__", "??"))
emb_model = load_e5()
gen_model = load_gemini()
conv = ConversationMulti()

###############################################################################
#Il server FastAPI è presente in questo file.
#Questo apre la porta 8080, accessibile dal browser su http://localhost:8080.
#uvicorn src.chatbot.peer_tutor_bot:app --host 0.0.0.0 --port 8080 --reload
###############################################################################
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
            fonti = "\n".join([f"- {s}" for s in used])
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

###############################################################################
#  API SIMULATION (FastAPI)
###############################################################################
app = FastAPI(title="PeerTutor RAG - Multiuser API")

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>ALEC- Peer Tutor Simulation</title>
        <style>
            body { font-family: sans-serif; max-width: 700px; margin: auto; padding: 20px; }
            textarea { width: 100%; height: 80px; }
            .log { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: auto; margin-bottom: 1em; }
            .alec { color: darkgreen; }
            .marcello { color: blue; }
            .davide { color: darkred; }
        </style>
    </head>
    <body>
        <h2>Simulazione PeerTutor (due studenti)</h2>
        <div class="log" id="log"></div>

        <select id="speaker">
            <option value="Davide">Davide</option>
            <option value="Marcello">Marcello</option>
        </select>
        <textarea id="msg" placeholder="Scrivi un messaggio..."></textarea><br/>
        <button onclick="send()">Invia</button>

        <script>
        async function send() {
            const speaker = document.getElementById('speaker').value;
            const message = document.getElementById('msg').value;
            const log = document.getElementById('log');
            if (!message.trim()) return;

            log.innerHTML += `<div class="${speaker.toLowerCase()}"><b>${speaker}:</b> ${message}</div>`;
            document.getElementById('msg').value = '';

            const resp = await fetch('/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speaker, message })
            });
            const data = await resp.json();
            if (data.message) {
                log.innerHTML += `<div class="alec"><b>ALEC → ${speaker}:</b> ${data.message}</div>`;
            } else {
                log.innerHTML += `<div><i>Errore:</i> ${JSON.stringify(data)}</div>`;
            }
            log.scrollTop = log.scrollHeight;
        }
        </script>
    </body>
    </html>
    """

@app.post("/message")
async def handle_message(request: Request):
    data = await request.json()
    speaker = data.get("speaker", "User")
    msg = data.get("message", "").strip()

    if not msg:
        return JSONResponse({"error": "Messaggio vuoto"}, status_code=400)

    # Comando /hint
    if msg.lower() == "/hint":
        store = conv.scaffold_for(speaker)
        nxt = store.pop_next_hint()
        if not nxt:
            return {"speaker": "Tutor", "message": f"Non ho altri indizi per {speaker}."}
        srcs = store.sources_block()
        if srcs:
            nxt += f"\n\nFonti:\n{srcs}"
        conv.add_turn(speaker, msg)
        conv.add_turn("Tutor", nxt)
        return {"speaker": "Tutor", "message": nxt}

    # Generazione risposta
    try:
        conv.scaffold_for(speaker).clear()
        reply = process_message(msg, index, mapping, emb_model, gen_model, conv, speaker)
        conv.add_turn(speaker, msg)
        conv.add_turn("Tutor", reply)
        return {"speaker": "Tutor", "message": reply}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
