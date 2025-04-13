import ast
import re
from langchain_ollama import OllamaLLM
import requests
from config import OLLAMA_API

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

def split_text_and_code(document: str):
    # Regex to capture fenced code blocks
    code_block_pattern = r"```(?:[\s\S]*?)```"
    
    # Split the document by the code pattern, capturing code blocks and text separately
    text = re.split(code_block_pattern, document)
    code = re.findall(code_block_pattern, document)
    # Iterate through the segments to separate text and code pairs
    return {"text": [t.strip() for t in text], "code": [c.strip() for c in code]}

def generate(model: str, prompt: str, context: list[dict], options: dict):
    """Generates a response using Ollama LLM.
    """
    # model = OllamaLLM(model=model)
    context_str = get_retrieved_data(context)
    query = f"""
    You are a helpful and friendly Next.js assistant. 
    Your responsibility is to answer user queries about Next.js. 
    Answer the question based only and only on the given context below (which got from Next.js documentation). If you can't answer the question, reply "I don't know".

    Context: {context_str}

    Question: {prompt}
    """
    # return model.invoke(query)
    # answer = requests.post("http://host.docker.internal:11434/api/generate", json={"model": model, "prompt": query, "stream": False})
    answer = requests.post(OLLAMA_API, json={"model": model, "prompt": query, "stream": False, "options": options})
    result = answer.json()
    result['retrieved_data'] = [ctx['title'] for ctx in context]
    return result
    
    
