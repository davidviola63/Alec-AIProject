from src.chatbot.app.routes import app

if __name__ == "__main__":
    import uvicorn
    print("✅ ALEC server avviato — visita http://localhost:8080")
    uvicorn.run("src.chatbot.start_alec:app", host="0.0.0.0", port=8080, reload=True)