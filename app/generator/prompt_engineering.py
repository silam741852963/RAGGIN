from fastapi import APIRouter, HTTPException
import logging
from app.classes.schemas import GeneratorRequest, HybridSearchRequest
from app.routes.data import load_supported_versions
from app.routes.search import hybrid_search
from config import SUPPORTED_VERSIONS_FILE
import ast

router = APIRouter()

def parse_code_content(code_content: str):
    parsed_data = ast.literal_eval(code_content)
    return parsed_data

def place_snippets_in_text(text_content: str, code_content_json: list) -> str:
    # return text_content + code_content_json
    code_template = "code_snippet_"
    text_content_add_snippets = text_content
    for i in range(len(code_content_json)):
        code_location = code_template + str(i+1)
        code_snippet = ""
        if code_content_json[i]['language']:
            code_snippet += code_content_json[i]['language']
        if code_content_json[i]['filename']:
            code_snippet += f" filename=\"{code_content_json[i]['filename']}\""
        if code_content_json[i]['switcher']:
            switcher = code_content_json[i]['switcher']
            if switcher:
                code_snippet += " switcher"
        code_snippet += "\n"
        code_snippet += code_content_json[i]['code']
        text_content_add_snippets = text_content_add_snippets.replace(code_location, code_snippet)
    return text_content_add_snippets

def get_retrieved_data(list_of_chunks: list[dict]) -> str:
    retrieved_data = ""

    for chunk in list_of_chunks:
        # print(chunk['code_content'])
        code_content_list = parse_code_content(chunk['code_content'])
        original_chunk = place_snippets_in_text(chunk['text_content'], code_content_list)
        retrieved_data += original_chunk
        retrieved_data += "\n\n"
    return retrieved_data

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
        hybrid_search_request = HybridSearchRequest(versionName=request.versionName, text_query=request.query)
        retrieved_docs = hybrid_search(hybrid_search_request)
        # prompt = f"Context: {retrieved_docs['results']}\n\nQuestion: {request.query}"
        # print(retrieved_docs)
        # return {"prompt": retrieved_docs}
        return {"prompt": request.query, "context": get_retrieved_data(retrieved_docs['results'])}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading versions file: {e}")