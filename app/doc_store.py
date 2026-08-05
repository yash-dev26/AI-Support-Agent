"""
Deliberately dumb document lookup.

This project is about agent orchestration and human-in-the-loop state
management, not retrieval quality — that's what Adaptive RAG already
demonstrates. Keyword scoring here is a feature, not a shortcut: it keeps
the two projects clearly differentiated on your resume/GitHub instead of
looking like the same RAG pipeline twice.

Docs are plain .txt/.md files dropped in /docs. Swap in a real embedding
call later only if you want to show you can decide when NOT to over-engineer.
"""
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"


def _load_docs() -> list[tuple[str, str]]:
    if not DOCS_DIR.exists():
        return []
    docs = []
    for path in DOCS_DIR.glob("*.*"):
        if path.suffix.lower() in (".txt", ".md"):
            docs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def search(query: str, top_k: int = 1) -> str:
    docs = _load_docs()
    if not docs:
        return "No company documents have been uploaded yet."

    query_terms = set(query.lower().split())
    scored = []
    for name, text in docs:
        # naive paragraph-level keyword overlap scoring
        best_para, best_score = "", 0
        for para in text.split("\n\n"):
            para_terms = set(para.lower().split())
            score = len(query_terms & para_terms)
            if score > best_score:
                best_score, best_para = score, para
        if best_score > 0:
            scored.append((best_score, name, best_para.strip()))

    if not scored:
        return "No relevant policy found in uploaded documents. Consider escalating to a human."

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]
    return "\n\n".join(f"[{name}]\n{para}" for _, name, para in top)
