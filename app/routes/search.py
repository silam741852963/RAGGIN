from fastapi import APIRouter, HTTPException
import logging
from app.models.schemas import HybridSearchRequest
from app.milvus.search_manager import HybridSearchManager

router = APIRouter()

@router.post("/search/hybrid")
def hybrid_search(request: HybridSearchRequest):
    """
    Performs a hybrid search over the Milvus collection using the provided parameters.
    Returns results with fields: title, metadata, version, textContent, codeContent, and tag.
    """
    try:
        search_manager = HybridSearchManager("nextjs_docs")
        results = search_manager.hybrid_search(
            versionName=request.versionName,
            sparseWeight=request.sparseWeight,
            denseTextWeight=request.denseTextWeight,
            denseCodeWeight=request.denseCodeWeight,
            topK=request.topK,
            filter_expr=request.filter,
            iterativeFilter=request.iterativeFilter,
            radius=request.radius,
            range_filter=request.rangeFilter
        )
        return {"results": results}
    except Exception as e:
        logging.exception("Error in hybrid_search endpoint")
        raise HTTPException(status_code=500, detail=str(e))