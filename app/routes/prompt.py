from __future__ import annotations

"""Prompt building & answer generation routes.

Separation of concerns:
* **_build_prompt_and_context()** – pure function that prepares the prompt
  and retrieval context.  No FastAPI types → easy unit‑test.
* **/prompt/enhance** – thin API wrapper that just returns that data.
* **/prompt/generate** – uses the same helper then calls Ollama.

Calling the helper directly from *generate* avoids an HTTP round‑trip, so
performance is already optimal; splitting the logic merely improves
readability & testability.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.classes.schemas import (
    PromptRequest,
    GeneratorRequest,
    SearchRequest,
    RetrieverOptions,
    GeneratorOptions,
    APIOptions,
    FileModel,
)
from app.routes.data import load_supported_versions
from app.routes.search import search
from utils import split_text_and_code, generate

router = APIRouter(prefix="/prompt", tags=["prompt"])
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# helper utilities
# -----------------------------------------------------------------------------


def _inline_files(text: str, files: List[FileModel] | None) -> str:
    """Append uploaded files to *text* as fenced blocks."""
    if not files:
        return text
    for f in files:
        text += f"\n```{f.file_extension} {f.file_name}\n{f.file_content}```"
    return text

import time

def _build_prompt_and_context(req: PromptRequest) -> dict[str, object]:
    """Core business logic used by both endpoints (no HTTP types)."""
    if req.version_name not in load_supported_versions():
        raise HTTPException(status_code=404, detail=f"Unsupported version '{req.version_name}'")

    text_parts = split_text_and_code(req.query)
    text_query = " ".join(text_parts["text"])
    code_query = " ".join(text_parts["code"])
    
    if req.file_list:
        for f in req.file_list:
            code_query += f"\n```{f.file_extension}\n{f.file_content}```"

    ropts = req.retriever_options or RetrieverOptions()
    search_req = SearchRequest(
        version_name=req.version_name,
        text_query=text_query,
        code_query=code_query,
        dense_code_weight=ropts.dense_code_weight,
        dense_text_weight=ropts.dense_text_weight,
        top_k=ropts.top_k or 3,
        filter_expr=ropts.filter_expr,
        iterative_filter=ropts.iterative_filter,
        radius_sparse=ropts.radius_sparse,
        range_sparse=ropts.range_sparse,
        radius_dense_text=ropts.radius_dense_text,
        range_dense_text=ropts.range_dense_text,
        radius_dense_code=ropts.radius_dense_code,
        range_dense_code=ropts.range_dense_code,
    )
    start_search = time.time()
    retrieved = search(search_req)
    end_search = time.time()
    print(f"Search time: {end_search - start_search}")
    prompt = _inline_files(req.query, req.file_list)
    return {"prompt": prompt, "context": retrieved["results"], "search_time": end_search - start_search}


# -----------------------------------------------------------------------------
# /prompt/enhance – return prompt + context only
# -----------------------------------------------------------------------------


@router.post("/enhance")
def enhance_prompt(req: PromptRequest):
    """API wrapper around `_build_prompt_and_context`."""
    try:
        return _build_prompt_and_context(req)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("enhance_prompt failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------------
# /prompt/generate – run Ollama
# -----------------------------------------------------------------------------


@router.post("/generate")
def generate_response(req: GeneratorRequest):
    """Compose prompt/context then invoke Ollama for the final answer."""
    try:
        api_opts: APIOptions = req.additional_options or APIOptions()
        retriever_opts = api_opts.retriever_options or RetrieverOptions()
        generator_opts = api_opts.generator_options or GeneratorOptions()

        prompt_ctx = _build_prompt_and_context(
            PromptRequest(
                version_name=req.version_name,
                query=req.query,
                file_list=req.file_list,
                retriever_options=retriever_opts,
                generator_options=generator_opts,
            )
        )
        return prompt_ctx
        # return generate(
        #     model=req.model,
        #     prompt=prompt_ctx["prompt"],
        #     context=prompt_ctx["context"],
        #     history=req.history or [],
        #     options=generator_opts.to_dict(),
        #     retrieved_time=prompt_ctx["search_time"],
        # )

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("generate_response failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.post("/test")
def test_generate(req: GeneratorRequest):
    print((req.model_dump_json()))
    return req