
# 🤖 ALEC — AI Learning for Education and Cooperation

**ALEC** è un prototipo di *peer-learning tutor* basato su tecniche di **Retrieval-Augmented Generation (RAG)** e integrato con il modello **Gemini Flash 2.5**.  
Il progetto è stato realizzato come MVP per il corso universitario di *Fondamenti di Intelligenza Artificiale*, con l’obiettivo di verificare la fattibilità di un sistema di tutoring intelligente, trasparente e multi-utente per l'attività di peer-learning. L'intero processo di sviluppo è documentato nel documento CRISP-DM presente nella cartella "CRISP-DM Document". Il progetto nasce dall'idea di offrire uno strumento efficace alla didattica dopo aver steso la Multivocal Literature Review incentrata sull'impatto educativo dei LLM, anch'essa presente nella cartella "Ricerca - MLR sull'impatto educativo del LLM".


## ⚙️ Installazione e configurazione del MVP

### 1. Clonare il repository
```bash
git clone https://github.com/davidviola63/Alec-AIProject.git
cd Alec-AIProject/Alec_Chatbot
````

### 2. Creare un ambiente virtuale

Si consiglia l’uso di `venv` o di un ambiente virtuale di PyCharm:

```bash
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Creare il file `.env` per la chiave API Gemini

Nella directory principale del progetto, crea un file chiamato `.env` con il seguente contenuto:

```
GEMINI_API_KEY=la_tua_chiave_personale
```

Per ottenere la chiave:

* visita [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
* accedi con un account Google
* copia la chiave API e incollala nel file `.env`


## 🚀 Avvio del chatbot

### Da **PyCharm** (modalità consigliata)

1. Apri il progetto in PyCharm.
2. Seleziona l’interprete Python associato al tuo ambiente virtuale( Progetto testato con Python 3.11).
3. Apri il file `start_alec.py`.
4. Clicca con il tasto destro → **Run 'start_alec'**.

### Da terminale

In alternativa, puoi avviare manualmente il server FastAPI:

```bash
uvicorn src.chatbot.start_alec:app --host 0.0.0.0 --port 8080 --reload
```



## 🌐 Accesso dal browser

Una volta avviato il server, apri nel browser:

```
http://localhost:8080
```

Vedrai l’interfaccia di chat di ALEC.
Da lì puoi interagire con il tutor utilizzando anche i comandi principali:

* `/hint` → Mostra un aiuto progressivo (scaffolding)
* `/stats` → Visualizza le statistiche personali e di sessione
* `/exit` → Chiude la sessione e genera il report finale (non reversibile)


## 📊 Stato del progetto

* **Tipo:** MVP (Minimum Viable Prototype)
* **Deployment:** Locale
* **Backend:** FastAPI + FAISS + Gemini Flash 2.5
* **Frontend:** HTML/CSS/JS minimale
* **Obiettivo:** Validare la fattibilità di implementare un tutor intelligente in una sessione di peer-learning offrendo uno stimolo didattico e personalizzato all'utente. Verificare la possibilità di implementare un modello che possa generare risposte mediante tecnologia RAG.

## 🧠 Autore

**Davide Viola**
Progetto sviluppato per il tirocinio universitario in Informatica — *Fondamenti di Intelligenza Artificiale*
Università degli Studi di Salerno


