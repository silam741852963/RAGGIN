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

from pydantic import BaseModel
from typing import Optional

class HybridSearchRequest(BaseModel):
    text_query: str
    code_query: str
    versionName: str
    sparseWeight: float = 1.0
    denseTextWeight: float = 1.0
    denseCodeWeight: float = 1.0
    topK: int = 10
    filter: Optional[str] = ""
    iterativeFilter: bool = False
    radius_sparse: float = 0.5
    range_sparse: float = 0.5
    radius_dense_text: float = 0.5
    range_dense_text: float = 0.5
    radius_dense_code: float = 0.5
    range_dense_code: float = 0.5

class RetrieveRequest(BaseModel):
    versionName: str
    m_text: Optional[int] = None
    ef_text: Optional[int] = None
    m_code: Optional[int] = None
    ef_code: Optional[int] = None
