from __future__ import annotations

"""Utility helpers used across the RAG‑in‑Next.js project.
    from utils import normalize_distance, generate
"""

from typing import List
import ast
import math
import re
import requests

from config import OLLAMA_API
from app.classes.schemas import ChatHistory

__all__ = [
    # distance
    "normalize_distance",
    # prompt helpers
    "parse_code_content",
    "place_snippets_in_text",
    "get_retrieved_data",
    "split_text_and_code",
    "history_string",
    "generate",
]

# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------

def normalize_distance(distance: float) -> float:
    """Map a raw similarity **distance** to [0, 1].

    0 → 1 (closest / highest relevance), +∞ → 0.
    Less‑than‑zero distances are clamped to 1.
    """
    val = (math.pi / 2 - math.atan(distance)) / (math.pi / 2)
    return max(0.0, min(val, 1.0))


# -----------------------------------------------------------------------------
# Markdown / snippet helpers
# -----------------------------------------------------------------------------

def parse_code_content(code_content: str):
    """Parse the JSON‑ish *code_content* column back to native Python."""
    return ast.literal_eval(code_content)


def place_snippets_in_text(text_content: str, code_content_json: list) -> str:
    """Replace placeholder markers (```code_snippet_N```) with real code blocks."""
    code_template = "code_snippet_"
    updated = text_content
    for idx, snippet in enumerate(code_content_json, start=1):
        marker = f"{code_template}{idx}"
        header_parts: List[str] = []
        if snippet["language"]:
            header_parts.append(snippet["language"])
        if snippet["filename"]:
            header_parts.append(f'filename="{snippet["filename"]}"')
        if snippet.get("switcher"):
            header_parts.append("switcher")
        replacement = " ".join(header_parts) + "\n" + snippet["code"]
        updated = updated.replace(marker, replacement)
    return updated


def get_retrieved_data(chunks: List[dict]) -> str:
    """Re‑assemble original docs (text + code) into a single context string."""
    data = []
    for chunk in chunks:
        code_list = parse_code_content(chunk["code_content"])
        combined = place_snippets_in_text(chunk["text_content"], code_list)
        data.append(combined)
    return "\n\n".join(data)


def split_text_and_code(document: str):  # noqa: D401 – util splitter
    """Return separate `text` / `code` lists from a markdown *document*."""
    pattern = r"```(?:[\s\S]*?)```"  # fenced blocks
    text_parts = re.split(pattern, document)
    code_parts = re.findall(pattern, document)
    return {"text": [t.strip() for t in text_parts], "code": [c.strip() for c in code_parts]}


def history_string(history: List[ChatHistory]) -> str:
    """Serialize past chat exchanges to a simple bracketed string block."""
    items = [f"<query>{h.query}</query> <response>{h.response}</response>" for h in history]
    return "<chat history>[\\n" + "\\n".join(items) + "]</chat history>"


# -----------------------------------------------------------------------------
# LLM generation helper
# -----------------------------------------------------------------------------

def _get_reference(context: List[dict]):
    """
    Extract related links from a list of context items.
    """
    base_url = "https://nextjs.org/docs/"
    res = []
    for c in context:
        # if c["title"]:
        #     res.append(c["title"])
        try:
            if c['related']:
                links = c['related']['link']
                if isinstance(links, list):
                    res.extend(c['related']['link'])
                else:
                    res.append(c['related']['link'])
        except Exception:
            pass
    # return [f"{base_url}{link}" for link in res] # + [c['title'] for c in context]\
    return {
        "links": [f"{base_url}{link}" for link in res],
        "docs": [c['title'] for c in context]
    }
    
import time

def generate(
    model: str,
    prompt: str,
    context: List[dict],
    history: List[ChatHistory],
    options: dict | None = None,
):
    """Query the Ollama API and return its JSON response augmented with titles."""
    context_str = history_string(history) + get_retrieved_data(context)
    full_prompt = f"""
You are a helpful and friendly Next.js assistant.
Answer **only** with information grounded in the context below. If unsure, reply "I don't know".

Context:\n{context_str}

Question:\n{prompt}
"""

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options
    start_generate = time.time()
    resp = requests.post(OLLAMA_API, json=payload, timeout=600)
    end_generate = time.time()
    print(f"Generate time: {end_generate - start_generate}")
    resp.raise_for_status()
    data = resp.json()
    data["retrieved_data"] = _get_reference(context=context)
    return data