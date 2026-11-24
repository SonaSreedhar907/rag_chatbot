"""Retrieve relevant chunks and ask the LLM to answer only from them.
CLI test:  python -m app.chat "How do I reset a Dinstar gateway?"
"""
import sys
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from . import config

_model = None
_col = None

SYSTEM = (
    "You are NETON Technologies' internal support assistant. "
    "Answer ONLY from the CONTEXT given. Be short and practical (steps as a numbered list). "
    "After each fact, cite the source like [file p.X]. "
    "If the answer is not in the context, reply exactly: "
    "\"I couldn't find this in our documents. Please contact the NETON team.\""
)
NOT_FOUND = "I couldn't find this in our documents. Please contact the NETON team."


def _load():
    global _model, _col
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
        _col = chromadb.PersistentClient(path=str(config.DB_DIR)).get_collection(config.COLLECTION)
    return _model, _col


def retrieve(question: str, brand: str | None = None, k: int = config.TOP_K):
    model, col = _load()
    vec = model.encode([question], normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=vec,
        n_results=k,
        where={"brand": brand} if brand else None,
    )
    return [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def call_llm(system: str, messages: list[dict]) -> str:
    if config.LLM_PROVIDER == "claude":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=config.CLAUDE_MODEL, max_tokens=800, system=system, messages=messages
        )
        return r.content[0].text
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={"model": config.OLLAMA_MODEL, "stream": False,
              "messages": [{"role": "system", "content": system}] + messages},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def answer(question: str, brand: str | None = None, history: list[dict] | None = None) -> dict:
    hits = retrieve(question, brand)
    hits = [h for h in hits if h["distance"] <= config.MAX_DISTANCE]
    if not hits:  # nothing close enough -> refuse without calling the LLM
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(
        f"[{h['meta']['file']} p.{h['meta']['page']}]\n{h['text']}" for h in hits
    )
    messages = (history or [])[-6:] + [
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    ]
    text = call_llm(SYSTEM, messages)
    sources = sorted({f"{h['meta']['file']} p.{h['meta']['page']}" for h in hits})
    return {"answer": text, "sources": sources}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I reset the gateway?"
    out = answer(q)
    print(out["answer"], "\n\nSources:", ", ".join(out["sources"]))
