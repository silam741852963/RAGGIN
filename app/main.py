# app/main.py
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pymilvus import connections
from huggingface_hub import snapshot_download

from config import MILVUS_URI, MODEL_CACHE_DIR
from app.routes import data, version, search, prompt

# ------------------------------------------------------------------#
#  Logging
# ------------------------------------------------------------------#
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
# silence Hugging Face + urllib3 DEBUG noise
for noisy in ("filelock", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.INFO)

log = logging.getLogger("raggin.main")

HF_REPO = "BAAI/bge-m3"
REV_FILE = ".snapshot_complete"


def snapshot_complete(cache_dir: Path) -> bool:
    """Check cache dir contains a completed snapshot."""
    return (cache_dir / REV_FILE).is_file()


def ensure_bge_m3(cache_dir: Path = Path(MODEL_CACHE_DIR)) -> None:
    """Download BGE‑M3 once; touch REV_FILE to mark completion."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_complete(cache_dir):
        log.info("✓ BGE‑M3 snapshot already present -> skip download")
        return

    log.info("Downloading BGE‑M3 model to %s …", cache_dir)

    # simply drop allow_patterns
    snapshot_download(
        repo_id=HF_REPO,
        cache_dir=cache_dir,
        resume_download=True,
    )

    # mark success
    (cache_dir / REV_FILE).touch()
    log.info("✓ Model download finished")


# ------------------------------------------------------------------#
#  Lifespan: run once on startup / shutdown
# ------------------------------------------------------------------#
@asynccontextmanager
async def lifespan(_: FastAPI):
    # heavy I/O here ⤵
    ensure_bge_m3()
    connections.connect(uri=MILVUS_URI)
    log.debug("Connected to Milvus at %s", MILVUS_URI)
    yield
    # close Milvus if you like:
    connections.disconnect()
    log.debug("Milvus connection closed")


app = FastAPI(title="RAGGIN", version="0.2", lifespan=lifespan)

# Routers
app.include_router(data.router)
app.include_router(version.router)
app.include_router(search.router)
app.include_router(prompt.router)

# ------------------------------------------------------------------#
#  Simple health routes
# ------------------------------------------------------------------#
@app.get("/")
def root():
    return {"message": "Connected to RAGGIN API!"}


# ------------------------------------------------------------------#
#  Dev entry‑point
# ------------------------------------------------------------------#
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)