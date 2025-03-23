import os

# PROJECT_ROOT is defined as the directory one level up from the config folder.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOWNLOADS_DIR = "/downloads"
SUPPORTED_VERSIONS_FILE = os.path.join("config", "supported_versions.txt")

# Constant for the dense vector dimension.
DENSE_VECTOR_DIM = 1024

MILVUS_URI = os.getenv("MILVUS_URI", "http://standalone:19530")

MODEL_CACHE_DIR="/BAAI/bge-m3"

OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
