from fastapi import APIRouter, HTTPException
import logging
from app.classes.schemas import PromptRequest, GeneratorRequest, HybridSearchRequest
from app.routes.data import load_supported_versions
from app.routes.search import hybrid_search
from config import SUPPORTED_VERSIONS_FILE
from app.generator.prompt_utils import get_retrieved_data, parse_code_content, place_snippets_in_text, split_text_and_code, generate

router = APIRouter()

@router.post("/enhance_prompt")
def enhance_prompt(request: PromptRequest):
    """
    Generates a response based on the provided query and versionName.
    """
    try:
        supported_versions = load_supported_versions()
        if request.versionName not in supported_versions:
            raise HTTPException(status_code=404, detail=f"Version {request.versionName} is not supported.")
        query_dict = split_text_and_code(request.query)
        text_query = " ".join(query_dict['text'])
        code_query = " ".join(query_dict['code'])
        hybrid_search_request = HybridSearchRequest(versionName=request.versionName, text_query=text_query, code_query=code_query)
        retrieved_docs = hybrid_search(hybrid_search_request)

        return {"prompt": request.query, "context": retrieved_docs['results']}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    
@router.post("/generate_response")
def generate_response(request: GeneratorRequest):
    """
    Generates a response based on the provided query and versionName.
    
    Example request:
    ```
    {
        "versionName": "v15.1.2",
        "query": "How to create new page?",
        "model": "llama3.2:3b"
    }
    ```
    """
    try:
        prompt = enhance_prompt(request)
        # return {"model": request.model, "prompt": prompt['prompt'], "context": prompt['context']}
        answer = generate(model=request.model, prompt=prompt['prompt'], context=prompt['context'])
        # return {"model": request.model, "answer": answer}
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)