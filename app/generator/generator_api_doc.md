## Generator
### Generate Response
```
POST /generate_response
```
- **Note:** The generator takes a user query along with retrieved documents from the retriever and constructs a context-aware prompt. The prompt is then sent to a locally hosted LLM via the [Ollama](https://ollama.com/) API.
- **Input:**
    - `versionName`: string (the chosen Next.js version)
    - `query`: string (the original question from the user)
    - `model`: string (Ollama model to use, e.g., `"llama3.2:3b"`, `"qwen:1.8b"`)
    - `file_list` (Optional): a list of FileModel
        - A FileModel consist of `fileName`, `FileExtension` and `FileContent`, all in type string
- **Output:** The response is returned as either:
    - **Single JSON Object** (for standard responses)
    - **Stream of JSON Objects** (for real-time generation)

    Each response contains the following fields:
    - `model`: string (the name of the Ollama model used)
    - `response`: string (the generated text from the LLM)
    - `retrieved_data`: list of retrieved documents
- **Processing:**
    1. Use **prompt** and **versionName** to retrieve relevance documents.
    2. Construct an **enhanced_prompt** using the original prompt with the retrieved documents.
    3. Send a **POST request** to the Ollama API endpoint: `POST http://localhost:11434/api/generate` (for Docker, send to `POST http://host.docker.internal:11434/api/generate`).
    4. Receive and return the generated response.
- **Example Request Payload:**
    ```json
    {
        "versionName": "v15.0.0",
        "query": "How does Next.js handle static generation?",
        "model": "llama3.2:3b",
        "file_list": [
            {
                "fileExtension": "js",
                "fileName": "next.config.js",
                "fileContent": "module.exports = { ... }"
            }
        ],
        "additional_options": {
            "retriever_options": {
                "denseCodeWeight": 0.5,
                "denseTextWeight": 0.5,
                "topK": 3,
            },
            "generator_options": {
                "temperature": 0.5,
                "top_p": 0.9
            }
        }
    }
    ```
- **Example Output:**
    ```json
    {
        "model": "llama3.2:3b",
        "response": "Next.js supports static site generation using `getStaticProps`. This function allows pre-rendering pages at build time for better performance.",
        "retrieved_data": [
            {document1},
            {document2},
            {document3}
        ]
    }
    ```
### **Why Ollama?**
- **Local execution**: Runs entirely on the user's machine, ensuring privacy.
- **Model flexibility**: Supports multiple models like `Qwen`, `Mistral`, and `Llama`.
- **Efficient API integration**: Simple HTTP-based API for quick inference.

