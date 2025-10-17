from __future__ import annotations
import time
from src.chatbot.script import retrieval, gemini_client
from src.chatbot.script.conversation_multi import ConversationMulti
from src.chatbot.multi_peer_tutor_bot import process_message

def run_multisession():
    print("PeerTutor RAG · Sessione Peer Learning (due studenti)")
    print("Comandi: /hint per indizi, /exit per uscire.\n")

    # Setup core
    index, mapping = retrieval.load_index_and_mapping()
    emb_model = retrieval.load_e5()
    gen_model = gemini_client.load_gemini()
    conv = ConversationMulti()

    participants = ["Davide", "Alessandra"]

    while True:
        for speaker in participants:
            try:
                msg = input(f"\n{speaker}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSessione terminata.")
                return

            if not msg:
                continue
            if msg.lower() in {"/exit", "/quit"}:
                print("Chiusura sessione.")
                return

            # Gestione /hint per speaker
            if msg.lower() == "/hint":
                store = conv.scaffold_for(speaker)
                nxt = store.pop_next_hint()
                if not nxt:
                    print(f"\nAlec → {speaker}: Non ho altri indizi per te.")
                    continue
                srcs = store.sources_block()
                if srcs:
                    nxt += f"\n\nFonti:\n{srcs}"
                print(f"\nAlec → {speaker}: {nxt}")
                conv.add_turn(speaker, msg)
                conv.add_turn("Tutor", nxt)
                continue

            # Reset scaffolding per chi parla
            conv.scaffold_for(speaker).clear()

            # Genera risposta tutor
            try:
                reply = process_message(
                    user_msg=msg,
                    index=index,
                    mapping=mapping,
                    emb_model=emb_model,
                    gen_model=gen_model,
                    conv=conv
                )
            except RuntimeError as e:
                print(f"[RateLimit] {e}"); time.sleep(6); continue
            except Exception as e:
                print(f"[ERRORE] {e}")
                continue

            print(f"\nAlec → {speaker}: {reply}")

            # Aggiorna storia condivisa
            conv.add_turn(speaker, msg)
            conv.add_turn("Tutor", reply)
