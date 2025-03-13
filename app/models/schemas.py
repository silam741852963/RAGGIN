from pydantic import BaseModel
from typing import Optional

class VersionsResponseItem(BaseModel):
    versionName: str
    downloaded: bool

class RetrieveRequest(BaseModel):
    versionName: str

class SchemaRequest(BaseModel):
    collection_name: str = "nextjs_docs"
    csv_file_path: str
    batch_size: int = 100
    uri: str = "http://0.0.0.0:19530"

class HybridSearchRequest(BaseModel):
    versionName: str
    sparseWeight: float
    denseTextWeight: float
    denseCodeWeight: float
    topK: int
    filter: str
    iterativeFilter: bool
    radius: float
    rangeFilter: float

class RetrieveRequest(BaseModel):
    versionName: str
    m_text: Optional[int] = None
    ef_text: Optional[int] = None
    m_code: Optional[int] = None
    ef_code: Optional[int] = None

class GeneratorRequest(BaseModel):
    versionName: str
    query: str