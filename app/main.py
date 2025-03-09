from fastapi import FastAPI
from app.routes import data, search, version_manager

app = FastAPI(title="RAGGIN", version="0.1")

# Include routers from different modules.
app.include_router(data.router)
app.include_router(version_manager.router)
app.include_router(search.router)

@app.get("/")
def read_root():
    return {"message": "Connected to RAGGIN API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)