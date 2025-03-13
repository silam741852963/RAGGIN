from fastapi import FastAPI
from pymilvus import connections
from config import MILVUS_URI
from app.routes import data, search, version

app = FastAPI(title="RAGGIN", version="0.2")

# Connect to Milvus once at application startup.
connections.connect(uri=MILVUS_URI)

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