from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent          #  …/<project>
CONFIG_DIR   = PROJECT_ROOT / "config"

# --------------------------------------------------------------------------- #
# data and model paths – overridable from the host env or docker‑compose
# --------------------------------------------------------------------------- #

DOWNLOADS_DIR: Path = Path(
    os.getenv("DOWNLOADS_DIR", PROJECT_ROOT / "downloads")
).resolve()

SUPPORTED_VERSIONS_FILE: Path = Path(
    os.getenv("SUPPORTED_VERSIONS_FILE", CONFIG_DIR / "supported_versions.txt")
).resolve()

MODEL_CACHE_DIR: Path = Path(
    os.getenv("MODEL_CACHE_DIR", PROJECT_ROOT / "models" / "bge-m3")
).resolve()

# --------------------------------------------------------------------------- #
# infrastructure
# --------------------------------------------------------------------------- #

# Milvus
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://standalone:19530")

# Ollama
OLLAMA_API: str = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

DENSE_VECTOR_DIM: int = 1024