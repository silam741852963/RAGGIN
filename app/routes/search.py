from fastapi import APIRouter, HTTPException
import logging
from app.classes.schemas import HybridSearchRequest
from app.milvus.search_manager import HybridSearchManager
from config import MILVUS_URI

router = APIRouter()

@router.post("/search/hybrid")
def hybrid_search(request: HybridSearchRequest):
    """
    Performs a hybrid search over the Milvus collection using the provided parameters.
    Returns results with fields: title, metadata, version, textContent, codeContent, and tag.
    """
    try:
        search_manager = HybridSearchManager("nextjs_docs", uri=MILVUS_URI)
        results = search_manager.hybrid_search(
            text_query=request.text_query,
            code_query=request.code_query,
            versionName=request.versionName,
            sparseWeight=request.sparseWeight,
            denseTextWeight=request.denseTextWeight,
            denseCodeWeight=request.denseCodeWeight,
            topK=request.topK,
            filter_expr=request.filter_expr,
            iterativeFilter=request.iterativeFilter,
            radius_sparse=request.radius_sparse,
            range_sparse=request.range_sparse,
            radius_dense_text=request.radius_dense_text,
            range_dense_text=request.range_dense_text,
            radius_dense_code=request.radius_dense_code,
            range_dense_code=request.range_dense_code,
        )
        return {"results": results}
    except Exception as e:
        logging.exception("Error in hybrid_search endpoint")
        raise HTTPException(status_code=500, detail=str(e))