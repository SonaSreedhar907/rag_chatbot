import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
DB_DIR = ROOT / "vectordb"
COLLECTION = "neton_docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # use a multilingual model if you need Arabic
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6
MAX_DISTANCE = 0.65  # cosine distance; above this = "not found in docs". Tune it in step 7.

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
