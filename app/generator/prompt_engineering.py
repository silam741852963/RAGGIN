from fastapi import APIRouter, HTTPException
import logging
from app.classes.schemas import PromptRequest, GeneratorRequest, HybridSearchRequest, RetrieverOptions, GeneratorOptions, APIOptions
from app.routes.data import load_supported_versions
from app.routes.search import hybrid_search
from config import SUPPORTED_VERSIONS_FILE
from app.generator.prompt_utils import get_retrieved_data, parse_code_content, place_snippets_in_text, split_text_and_code, generate, history_string

router = APIRouter()

@router.post("/test_prompt")
def test_prompt(request: PromptRequest):
    try:
        if request.file_list is not None:
            for file in request.file_list:
                request.query += f"```{file.fileExtension} {file.fileName}\n{file.fileContent}```"
        return {"prompt": request.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

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
        denseCodeWeight = request.retriever_options.denseCodeWeight if request.retriever_options.denseCodeWeight is not None else 0.5
        denseTextWeight = request.retriever_options.denseTextWeight if request.retriever_options.denseTextWeight is not None else 0.5
        topK = request.retriever_options.topK if request.retriever_options.topK is not None else 3
        filter_expr = request.retriever_options.filter_expr if request.retriever_options.filter_expr is not None else None
        iterativeFilter = request.retriever_options.iterativeFilter if request.retriever_options.iterativeFilter is not None else False
        radius_sparse = request.retriever_options.radius_sparse if request.retriever_options.radius_sparse is not None else 0.5
        range_sparse = request.retriever_options.range_sparse if request.retriever_options.range_sparse is not None else 0.5
        radius_dense_text = request.retriever_options.radius_dense_text if request.retriever_options.radius_dense_text is not None else 0.5
        range_dense_text = request.retriever_options.range_dense_text if request.retriever_options.range_dense_text is not None else 0.5
        radius_dense_code = request.retriever_options.radius_dense_code if request.retriever_options.radius_dense_code is not None else 0.5
        range_dense_code = request.retriever_options.range_dense_code if request.retriever_options.range_dense_code is not None else 0.5
        
        hybrid_search_request = HybridSearchRequest(versionName=request.versionName,
                                                    text_query=text_query, 
                                                    code_query=code_query,
                                                    denseCodeWeight=denseCodeWeight,
                                                    denseTextWeight=denseTextWeight,
                                                    topK=topK,
                                                    filter_expr=filter_expr,
                                                    iterativeFilter=iterativeFilter,
                                                    radius_sparse=radius_sparse,
                                                    range_sparse=range_sparse,
                                                    radius_dense_text=radius_dense_text,
                                                    range_dense_text=range_dense_text,
                                                    radius_dense_code=radius_dense_code,
                                                    range_dense_code=range_dense_code,
                                                    )
        retrieved_docs = hybrid_search(hybrid_search_request)
        final_query = request.query
        if request.file_list is not None:
            for file in request.file_list:
                final_query += f"\n```{file.fileExtension} {file.fileName}\n{file.fileContent}```"
        # history = history_string(request.history)
        return {"prompt": final_query, "context": retrieved_docs['results']}
            
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
        "model": "llama3.2:3b",
        "file_list": [
            {
                "fileExtension": "js",
                "fileName": "next.config.js",
                "fileContent": "module.exports = { ... }"
            },
            {
                "fileExtension": "js",
                "fileName": "index.js",
                "fileContent": "export default function Home() { ... }"
            }
        ],
        "additional_options": {
            "retriever_options": {
                "denseCodeWeight": 0.5,
                "denseTextWeight": 0.5,
                "topK": 3
            },
            "generator_options": {
                "temperature": 0.5,
                "top_p": 0.9
            }
        }
    }
    ```
    """
    try:
        if request.additional_options is None:
            request.additional_options = APIOptions()
        retriever_options = request.additional_options.retriever_options if request.additional_options.retriever_options is not None else RetrieverOptions()
        generator_options = request.additional_options.generator_options if request.additional_options.generator_options is not None else GeneratorOptions()
        # return {"generator_options": generator_options, "retriever_options": retriever_options}
        chat_history = request.history if request.history is not None else []
        prompt_request = PromptRequest(versionName=request.versionName, 
                                       query=request.query,
                                       file_list=request.file_list, 
                                       retriever_options=retriever_options, 
                                       generator_options=generator_options)
        prompt = enhance_prompt(prompt_request)
        # return {"model": request.model, "prompt": prompt['prompt'], "context": prompt['context']}
        # g_option = request.additional_options.generator_options if request.additional_options.generator_options is not None else dict()
        # return {"model": request.model, "prompt": prompt['prompt'], "context": prompt['context'], "history": chat_history, "generator_options": generator_options}
        # return {"options": generator_options.get_dict()}
        answer = generate(model=request.model, prompt=prompt['prompt'], context=prompt['context'], history=chat_history, options=generator_options.get_dict())
        # return {"model": request.model, "answer": answer}
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)