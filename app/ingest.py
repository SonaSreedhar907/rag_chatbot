"""Load documents -> chunk -> embed -> store in ChromaDB.
Run:  python -m app.ingest
Folder name under data/raw/ becomes the 'brand' tag (wildix, dinstar, ...).
"""
import shutil
import pandas as pd
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from . import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def load_pdf(path):
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            yield page_no, text


def load_table(path):
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    for row_no, row in df.iterrows():
        text = "; ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v))
        if text.strip():
            yield row_no + 2, text  # +2 = spreadsheet row number (header is row 1)


def load_text(path):
    yield 1, path.read_text(errors="ignore")


def iter_chunks():
    for path in sorted(config.RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".pdf":
            pages, whole_row = load_pdf(path), False
        elif ext in (".csv", ".xlsx"):
            pages, whole_row = load_table(path), True  # one row = one chunk
        elif ext in (".txt", ".md"):
            pages, whole_row = load_text(path), False
        else:
            continue
        brand = path.relative_to(config.RAW_DIR).parts[0] if len(path.relative_to(config.RAW_DIR).parts) > 1 else "general"
        for page_no, text in pages:
            parts = [text] if whole_row else chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, part in enumerate(parts):
                yield {
                    "id": f"{path.name}:{page_no}:{i}",
                    "text": part,
                    "meta": {"brand": brand, "file": path.name, "page": page_no},
                }


def main():
    items = list(iter_chunks())
    if not items:
        print(f"No documents found in {config.RAW_DIR}. Add PDFs/CSVs and run again.")
        return
    print(f"{len(items)} chunks found. Loading embedding model...")
    model = SentenceTransformer(config.EMBED_MODEL)

    if config.DB_DIR.exists():
        shutil.rmtree(config.DB_DIR)  # simple: rebuild from scratch each run
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(items), 64):
        batch = items[i:i + 64]
        vectors = model.encode([b["text"] for b in batch], normalize_embeddings=True).tolist()
        col.add(
            ids=[b["id"] for b in batch],
            documents=[b["text"] for b in batch],
            metadatas=[b["meta"] for b in batch],
            embeddings=vectors,
        )
        print(f"  stored {min(i + 64, len(items))}/{len(items)}")
    print("Done. Chunks in DB:", col.count())


if __name__ == "__main__":
    main()
