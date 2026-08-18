"""
Tests for the Qdrant-backed vector store. Uses a real local Qdrant
instance (embedded mode, in a throwaway tmp_path) with a deterministic
fake embedding function monkeypatched in — no live OpenAI calls, but the
actual Qdrant read/write/search path is exercised for real, not mocked.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.services import vector_store

_VOCAB = ["refund", "refunds", "delivery", "shipping", "days", "business", "policy", "orders"]


def _fake_embed(text: str) -> list[float]:
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    vec = [1.0 if t in tokens else 0.0 for t in _VOCAB]
    vec += [0.0] * (vector_store.EMBED_DIM - len(vec))
    return vec


@pytest.fixture(autouse=True)
def isolated_qdrant(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "QDRANT_PATH", tmp_path / "qdrant_test")
    monkeypatch.setattr(vector_store, "_embed", _fake_embed)
    vector_store._client = None
    yield
    vector_store._client = None


def test_search_on_empty_collection_returns_empty_list():
    assert vector_store.search("refund policy") == []


def test_index_and_search_finds_relevant_paragraph():
    vector_store.index_document(
        "refund_policy.md",
        "Refunds are issued within 14 days of delivery.\n\nShipping takes 3-5 business days.",
    )
    results = vector_store.search("refund delivery", top_k=2)
    assert results
    assert results[0]["text"].startswith("Refunds are issued")
    assert results[0]["doc_name"] == "refund_policy.md"


def test_reindexing_same_content_is_idempotent():
    text = "Refunds are issued within 14 days of delivery.\n\nShipping takes 3-5 business days."
    vector_store.index_document("refund_policy.md", text)
    vector_store.index_document("refund_policy.md", text)

    client = vector_store._get_client()
    count = client.count(vector_store.COLLECTION).count
    assert count == 2, "re-indexing identical content should upsert in place, not duplicate"


def test_point_id_is_deterministic_for_same_content():
    id1 = vector_store._point_id("doc.md", "some paragraph")
    id2 = vector_store._point_id("doc.md", "some paragraph")
    assert id1 == id2


def test_point_id_differs_for_different_content():
    id1 = vector_store._point_id("doc.md", "paragraph one")
    id2 = vector_store._point_id("doc.md", "paragraph two")
    assert id1 != id2


def test_index_all_docs_indexes_every_txt_and_md_file(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("Refunds within 14 days.")
    (docs_dir / "b.txt").write_text("Shipping takes 3 business days.")
    (docs_dir / "ignore.json").write_text("{}")
    monkeypatch.setattr(vector_store, "DOCS_DIR", docs_dir)

    total = vector_store.index_all_docs()
    assert total == 2  # one paragraph each from a.md and b.txt, json ignored


