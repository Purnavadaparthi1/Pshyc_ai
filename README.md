# 🧠 PSYCH.AI

> AI-powered psychology tutor for IGNOU MA Psychology students  
> RAG-first · Gemini fallback agent · Production-ready · Single command startup

---

## Architecture

```
User Query
    │
    ▼
RAG Pipeline (ChromaDB + sentence-transformers)
    │
    ├── similarity ≥ 0.35 ──► Gemini (RAG-grounded response + IGNOU source citations)
    │
    └── similarity < 0.35 ──► Gemini Agent (pure psychology tutor fallback)
                                  (acts as expert psychiatrist/tutor, answers everything)
```

---

## File Structure

```
psych-ai/
├── start.sh                        # ← Single command: installs + builds + runs everything
│
├── backend/
│   ├── main.py                     # FastAPI app — serves API + built React frontend
│   ├── .env.example                # Copy to .env, add your GEMINI_API_KEY
│   ├── requirements.txt
│   │
│   ├── api/
│   │   └── chat.py                 # POST /api/chat  |  GET /api/health
│   │
│   ├── core/
│   │   ├── config.py               # Pydantic settings (reads .env)
│   │   ├── rag.py                  # ChromaDB query pipeline
│   │   └── agent.py                # gemini-2.5-flash agent (RAG + fallback)
│   │
│   ├── scripts/
│   │   └── ingest.py               # PDF/TXT → ChromaDB ingestion
│   │
│   └── data/
│       └── documents/              # ← Drop your IGNOU PDFs here
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js              # Dev proxy → :8000, build → dist/
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── PsychChatbot.jsx        # ← THE component (chatbot UI, all-in-one)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Gemini API key → https://aistudio.google.com/app/apikey

### Step 1 — Clone & configure
```bash
git clone <your-repo-url>
cd psych-ai

# Set your Gemini API key
cp backend/.env.example backend/.env
nano backend/.env          # set GEMINI_API_KEY=your_key_here
```

### Step 2 — (Optional but recommended) Ingest your IGNOU PDFs
```bash
# Drop your IGNOU course PDFs into:
cp ~/Downloads/MPC-001.pdf backend/data/documents/
cp ~/Downloads/MPC-007.pdf backend/data/documents/
# ... add as many as you have

# Run the ingestion script
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py
```

### Step 3 — Start everything
```bash
chmod +x start.sh
./start.sh
```

Open **http://localhost:8000** 🎉

That's it. The start script:
1. Creates a Python venv and installs dependencies
2. Runs `npm install` + `npm run build` in the frontend
3. Starts FastAPI which serves both the API and the built React app

---

## API Endpoints

### `POST /api/chat`
```json
// Request
{
  "message": "Explain Freud's id, ego, and superego",
  "history": [
    { "role": "user", "content": "What is psychoanalysis?" },
    { "role": "assistant", "content": "Psychoanalysis is..." }
  ]
}

// Response
{
  "reply": "Freud's structural model...",
  "source": "rag",              // "rag" | "gemini"
  "rag_sources": ["MPC-003.pdf"],
  "rag_similarity": 0.72
}
```

### `GET /api/health`
```json
{
  "status": "ok",
  "rag_chunks": 4821,
  "model": "gemini-2.5-flash"
}
```

---

## Ingestion Script

```bash
# Ingest everything in data/documents/
python scripts/ingest.py

# Ingest a single file
python scripts/ingest.py --file path/to/MPC-007.pdf

# Wipe the vector store and re-ingest fresh
python scripts/ingest.py --reset
```

Supported formats: `.pdf`, `.txt`, `.md`

---

## Development Mode (hot-reload)

Run backend and frontend separately for fast iteration:

```bash
# Terminal 1 — Backend with auto-reload
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend with HMR
cd frontend
npm run dev     # → http://localhost:5173 (proxies /api to :8000)
```

---

## Configuration (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | **required** | Your Google Gemini API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores its index |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RAG_SIMILARITY_THRESHOLD` | `0.35` | Below this → Gemini fallback |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |

---

## RAG Tuning Tips

| Symptom | Fix |
|---|---|
| Too many Gemini fallbacks | Lower `RAG_SIMILARITY_THRESHOLD` (try 0.25) |
| Irrelevant RAG results | Raise threshold (try 0.45) |
| Slow first response | Pre-warm singletons (already done in `lifespan`) |
| Large PDF slow to ingest | Increase `CHUNK_SIZE` in `ingest.py` |

---

## Enhancements (Planned)
- [ ] Voice input/output (Web Speech API + Google TTS)
- [ ] Study planner & exam prep module
- [ ] User authentication + session persistence (Redis)
- [ ] Progress tracking dashboard
- [ ] Multi-language support (Hindi)
- [ ] Weekly research paper ingestion pipeline
- [ ] Admin panel for document management
