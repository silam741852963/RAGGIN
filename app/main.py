import os
import logging
from fastapi import FastAPI
from pymilvus import connections
from config import MILVUS_URI, MODEL_CACHE_DIR
from app.routes import data, version, search
from huggingface_hub import snapshot_download


# Configure logging.
logging.basicConfig(level=logging.DEBUG)

# Check if the directory is empty.
if not os.listdir(MODEL_CACHE_DIR):  # returns an empty list if the directory is empty
    logging.info(f"Model directory {MODEL_CACHE_DIR} is empty. Downloading model files...")
    try:
        snapshot_download(repo_id="BAAI/bge-m3", cache_dir=MODEL_CACHE_DIR)
        logging.info("Model files downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading model files: {e}")
else:
    logging.info("Model files already downloaded in {MODEL_CACHE_DIR}.")

app = FastAPI(title="RAGGIN", version="0.2")

# Connect to Milvus once at application startup.
connections.connect(uri=MILVUS_URI)
logging.debug(f"Connected to Milvus at {MILVUS_URI}")

# Include routers from different modules.
app.include_router(data.router)
app.include_router(version.router)
app.include_router(search.router)

@app.get("/")
def read_root():
    return {"message": "Connected to RAGGIN API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)