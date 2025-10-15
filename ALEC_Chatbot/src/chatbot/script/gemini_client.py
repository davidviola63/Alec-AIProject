from __future__ import annotations
import os, json
from dotenv import load_dotenv
import google.generativeai as genai
from .config import ENV_PATH, GEMINI_MODEL
from .rate_limiter import RateLimiter
from .debug import DEBUG_JUDGE, DEBUG_SAVE, dbg

# istanza globale (iniettata dal runner)
RATE: RateLimiter | None = None

def load_gemini() -> genai.GenerativeModel:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(f"API key non trovata (GEMINI_API_KEY/GOOGLE_API_KEY in {ENV_PATH}).")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def gemini_json(model, prompt: str) -> dict:
    if RATE: RATE.check(1000)
    resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    if RATE: RATE.commit(1000)
    t = resp.text
    try:
        return json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end+1])
        raise

def gemini_text(model, prompt: str) -> str:
    if RATE: RATE.check(2000)
    resp = model.generate_content(prompt)
    if RATE: RATE.commit(2000)
    return resp.text.strip()
