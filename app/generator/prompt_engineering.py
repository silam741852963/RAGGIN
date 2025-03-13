from fastapi import APIRouter, HTTPException
import logging
from app.models.schemas import GeneratorRequest, HybridSearchRequest
from app.routes.data import load_supported_versions
from app.routes.search import hybrid_search
from config import SUPPORTED_VERSIONS_FILE

router = APIRouter()

@router.post("/enhance_prompt")
def enhance_prompt(request: GeneratorRequest):
    """
    Generates a response based on the provided query and versionName.
    """
    try:
        supported_versions = load_supported_versions()
        if request.versionName not in supported_versions:
            raise HTTPException(status_code=404, detail=f"Version {request.versionName} is not supported.")
        # Implement logic to generate a response based on the query and versionName
        # Return the generated response
        # hybrid_search_request = HybridSearchRequest(versionName=request.versionName, sparseWeight=0.5, denseTextWeight=0.5, denseCodeWeight=0.5, topK=5, filter="", iterativeFilter=False, radius=0.0, rangeFilter=0.0)
        # retrieved_docs = hybrid_search(hybrid_search_request)
        # prompt = f"Context: {retrieved_docs['results']}\n\nQuestion: {request.query}"
        # print(retrieved_docs)
        # return {"prompt": retrieved_docs}
        return {"prompt": request.query}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading versions file: {e}")