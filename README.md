# NETON RAG Support Bot (starter)

1. python -m venv .venv && source .venv/bin/activate   (Windows: .venv\Scripts\activate)
2. pip install -r requirements.txt
3. cp .env.example .env     (default = local Ollama; run: ollama pull llama3.1:8b)
4. Put PDFs / CSV / XLSX / TXT in data/raw/<brand>/   (folder name = brand filter)
5. python -m app.ingest      (re-run whenever documents change)
6. python -m app.chat "How do I reset a Dinstar gateway?"
7. uvicorn app.api:app --reload   -> http://localhost:8000

Switch to Claude: set LLM_PROVIDER=claude and ANTHROPIC_API_KEY in .env.
Tune MAX_DISTANCE / TOP_K / chunk sizes in app/config.py.
