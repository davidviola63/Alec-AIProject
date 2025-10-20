from src.chatbot.script.config.prompts import REPORT_INSTRUCTIONS
from src.chatbot.script.core.gemini_client import gemini_text


def generate_final_report(conv, gen_model):
    if not hasattr(conv, "analytics") or not conv.analytics.by_user:
        return "<p>Nessun dato disponibile per generare il report.</p>"

    payload = conv.analytics.by_user
    rows = []
    for user, stats in payload.items():
        s = stats.summary()
        rows.append(
            f"<li><b>{user}</b>: turni={s['turns_total']}, pertinenti={s['pertinenti']}, "
            f"qualità media={s['quality_mean']:.2f}, fonti principali={', '.join(s['top_sources']) or 'Nessuna'}</li>"
        )
    context_html = "<ul>" + "".join(rows) + "</ul>"

    prompt = (
        f"{REPORT_INSTRUCTIONS}\n\n"
        f"DATI SESSIONE:\n{context_html}\n\n"
        "Crea un report didattico sintetico e chiaro basato su questi dati."
    )

    report_body = gemini_text(gen_model, prompt)
    return f"""
    <!DOCTYPE html>
    <html lang='it'>
    <head>
        <meta charset='UTF-8'>
        <title>Report finale ALEC</title>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }}
            h1 {{ color: darkgreen; }}
            hr {{ margin: 1em 0; }}
            p, li {{ line-height: 1.5em; }}
            ul {{ margin-left: 1.2em; }}
        </style>
    </head>
    <body>
        <h1>📘 Report finale di sessione</h1>
        <hr>
        {report_body}
    </body>
    </html>
    """
