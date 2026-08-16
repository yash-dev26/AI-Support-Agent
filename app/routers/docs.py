import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends

from app.core.rate_limit import limiter
from app.core.auth import get_current_token, TokenPayload, require_support_agent
from app.services.vector_store import DOCS_DIR
from app.services import vector_store

logger = logging.getLogger("support_agent.docs")
router = APIRouter(prefix="/docs", tags=["docs"])


@router.post("/upload")
@limiter.limit("5/minute")
async def upload_doc(request: Request, file: UploadFile = File(...), current: TokenPayload = Depends(get_current_token)):
    """Upload a company doc (.txt/.md) and index it into the local Qdrant
    store so check_policy's RAG path can retrieve from it. Requires a
    support_agent token — this is an admin-ish action, not something a
    customer should be able to do.

    Rate limited (5/min, per remote address — no user_id on this route)
    since each upload triggers one embedding call per paragraph; nothing
    stops a large or repeated upload from running up embedding costs
    otherwise.
    """
    require_support_agent(current)

    if file.filename is None or not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported.")

    try:
        DOCS_DIR.mkdir(exist_ok=True)
        dest = DOCS_DIR / file.filename
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception:
        logger.exception("Failed to save uploaded doc %s", file.filename)
        raise HTTPException(status_code=500, detail="Could not save the uploaded document.")

    try:
        chunks_indexed = vector_store.index_document(file.filename, contents.decode("utf-8", errors="ignore"))
    except Exception:
        logger.exception("Saved %s but failed to index it into Qdrant", file.filename)
        # the file is saved either way — degrade rather than fail the whole
        # request, since the doc can still be picked up by index_all_docs
        # on next startup
        chunks_indexed = 0

    return {"status": "uploaded", "filename": file.filename, "chunks_indexed": chunks_indexed}
