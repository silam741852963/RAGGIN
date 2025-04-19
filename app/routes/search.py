import logging
from functools import lru_cache
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from app.classes.schemas import SearchRequest
from app.milvus.search_manager import SearchManager
from config import MILVUS_URI


# -----------------------------------------------------------------------------
# Lazy‑initialised manager (reuse across requests)
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_manager() -> SearchManager:
    """Create (or reuse) a single SearchManager instance."""
    logging.debug("Initialising SearchManager …")
    return SearchManager("nextjs_docs", uri=MILVUS_URI)


router = APIRouter()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.post("/search", response_model=Dict[str, Any])
def search(req: SearchRequest):
    """Run a hybrid (sparse + dense) search and return the merged top‑k results."""
    try:
        mgr = _get_manager()
        print(req.top_k)
        results = mgr.search(
            text_query=req.text_query,
            code_query=req.code_query,
            version=req.version_name,
            sparse_weight=req.sparse_weight,
            dense_text_weight=req.dense_text_weight,
            dense_code_weight=req.dense_code_weight,
            top_k=req.top_k,
            filter_expr=req.filter_expr,
            # radius / range tuning
            radius_sparse=req.radius_sparse,
            range_sparse=req.range_sparse,
            radius_dense_text=req.radius_dense_text,
            range_dense_text=req.range_dense_text,
            radius_dense_code=req.radius_dense_code,
            range_dense_code=req.range_dense_code,
        )
        return {"results": results}

    except ValueError as ve:
        # input validation errors propagated from manager
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        logging.exception("Search failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc