"""
Local persistent vector store for company policy docs, using Qdrant's
embedded mode (qdrant-client with path=..., no server process, no Docker
— just a local folder on disk). This is the persistence layer beneath
check_policy's RAG path.

Deliberately NOT the same shape as Adaptive RAG's retrieval stack: single
flat collection, no hybrid dense+sparse fusion, no reranking, no planner.
That's proportionate to what this project needs — see policy_engine.py's
docstring for the fuller reasoning on why retrieval here stays
intentionally thin. FAQ-style questions never reach this at all (see
faq.py), which keeps embedding calls limited to queries that actually
need semantic search.
"""
import hashlib
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
QDRANT_PATH = Path(__file__).parent.parent / "data" / "qdrant"
COLLECTION = "policy_docs"
EMBED_DIM = 1536  # text-embedding-3-small
EMBED_MODEL = "text-embedding-3-small"

_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        QDRANT_PATH.mkdir(exist_ok=True)
        _client = QdrantClient(path=str(QDRANT_PATH))
        if not _client.collection_exists(COLLECTION):
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    return _client


def _embed(text: str) -> list[float]:
    """Calls the OpenAI embeddings API. Kept as its own function so tests
    can monkeypatch it instead of hitting the network."""
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def _point_id(doc_name: str, paragraph: str) -> str:
    """Deterministic id from content hash, so re-indexing the same
    doc/paragraph upserts in place instead of duplicating on every restart
    or re-upload."""
    digest = hashlib.sha256(f"{doc_name}:{paragraph}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def index_document(doc_name: str, text: str) -> int:
    """Chunk by paragraph, embed each, upsert into the local Qdrant
    collection. Returns how many paragraphs were indexed."""
    client = _get_client()
    points = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        vector = _embed(para)
        points.append(PointStruct(
            id=_point_id(doc_name, para),
            vector=vector,
            payload={"doc_name": doc_name, "text": para},
        ))
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def index_all_docs() -> int:
    """Index every .txt/.md file currently in app/data/docs/. Safe to call
    repeatedly (e.g. on every app startup) — upserts are idempotent by
    content hash, so re-indexing an unchanged doc is a no-op in effect."""
    if not DOCS_DIR.exists():
        return 0
    total = 0
    for path in DOCS_DIR.glob("*.*"):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            total += index_document(path.name, text)
    return total


def search(query: str, top_k: int = 3) -> list[dict]:
    """Returns up to top_k {doc_name, text, score} matches, best first.
    Empty list if the collection has nothing indexed yet."""
    client = _get_client()
    if client.count(COLLECTION).count == 0:
        return []
    vector = _embed(query)
    result = client.query_points(collection_name=COLLECTION, query=vector, limit=top_k)
    return [
        {"doc_name": p.payload["doc_name"], "text": p.payload["text"], "score": p.score}
        for p in result.points
    ]


