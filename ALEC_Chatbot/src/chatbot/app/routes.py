# src/chatbot/app/routes.py
import os
from fastapi import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


from src.chatbot.script.config.variables import REPORT_DIR
from src.chatbot.script.core.process_message import process_message
from src.chatbot.script.rf6.report_generator import generate_final_report
from src.chatbot.script.rf4.conversation_multi import ConversationMulti
from src.chatbot.script.rf1.retrieval import load_index_and_mapping, load_e5
from src.chatbot.script.core.gemini_client import load_gemini

# Crea app
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inizializzazione risorse
index, mapping = load_index_and_mapping()
emb_model = load_e5()
gen_model = load_gemini()
conv = ConversationMulti()

# === ROUTE HOMEPAGE ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "homepage.html")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve la homepage HTML dell'applicazione."""
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<h1>Errore interno:</h1><pre>{e}</pre>", status_code=500)
# === ROUTE MESSAGE ===
@app.post("/message")
async def handle_message(request: Request):
    data = await request.json()
    speaker = data.get("speaker", "User")
    msg = data.get("message", "").strip()

    if not msg:
        return JSONResponse({"error": "Messaggio vuoto"}, status_code=400)

    if conv.is_closed():
        return {
            "speaker": "Tutor",
            "message": "<p><i>La sessione è terminata. Non è più possibile inviare messaggi.</i></p>",
        }

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

        if hasattr(conv, "analytics"):
            conv.analytics.add_hint(speaker)

        conv.add_turn("Tutor", nxt)
        return {"speaker": "Tutor", "message": nxt}

    #Comando /stats
    if msg.lower() == "/stats":
        if hasattr(conv, "analytics"):
            html = conv.analytics.get_stats_html(speaker)
            conv.add_turn(speaker, msg)
            conv.add_turn("Tutor", html)
            return {"speaker": "Tutor", "message": html}

    if msg.lower() == "/exit":
        report_html = generate_final_report(conv, gen_model)

        report_path = os.path.join(REPORT_DIR, f"report_{conv.analytics.session_id}.html")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        conv.close_session()

        return {
            "speaker": "Tutor",
            "message": (
                f"Report finale generato.<br>"
                f"<a href='/report/{conv.analytics.session_id}' target='_blank'>Apri report</a>"
            ),
        }
    # Generazione risposta
    try:
        conv.scaffold_for(speaker).clear()
        reply = process_message(msg, index, mapping, emb_model, gen_model, conv, speaker)
        conv.add_turn(speaker, msg)
        conv.add_turn("Tutor", reply)
        return {"speaker": "Tutor", "message": reply}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === ROUTE REPORT ===
# === ROUTE REPORT (parametrica) ===
@app.get("/report/{session_id}", response_class=HTMLResponse)
async def show_report(session_id: str = Path(..., description="ID della sessione")):
    """Mostra il report finale della sessione specificata."""
    try:
        report_path = os.path.join(REPORT_DIR, f"report_{session_id}.html")
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(f"<p>Nessun report trovato per la sessione {session_id}.</p>")

