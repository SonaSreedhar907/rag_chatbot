"""Web API + tiny chat page.   Run: uvicorn app.api:app --reload
Open http://localhost:8000
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from . import chat

app = FastAPI(title="NETON Support Bot")


class Ask(BaseModel):
    question: str
    brand: str | None = None
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def ask(body: Ask):
    return chat.answer(body.question, body.brand, body.history)


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text()
