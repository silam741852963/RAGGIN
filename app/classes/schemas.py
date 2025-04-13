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
    uri: str = "http://standalone:19530"

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
    filter_expr: Optional[str] = ""
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

class FileModel(BaseModel):
    fileName: str = ""
    fileExtension: str = ""
    fileContent: str
    
class RetrieverOptions(BaseModel):
    m_text: Optional[int] = None
    ef_text: Optional[int] = None
    m_code: Optional[int] = None
    ef_code: Optional[int] = None
    sparseWeight: Optional[float] = 1.0
    denseTextWeight: Optional[float] = 1.0
    denseCodeWeight: Optional[float] = 1.0
    topK: Optional[int] = None
    filter_expr: Optional[str] = None
    iterativeFilter: Optional[bool] = False
    radius_sparse: Optional[float] = 0.5
    range_sparse: Optional[float] = 0.5
    radius_dense_text: Optional[float] = 0.5
    range_dense_text: Optional[float] = 0.5
    radius_dense_code: Optional[float] = 0.5
    range_dense_code: Optional[float] = 0.5
    
class GeneratorOptions(BaseModel):
    microstat: Optional[int] = None
    microstat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[str] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    
    def get_dict(self) -> dict:
        return {
            "microstat": self.microstat,
            "microstat_eta": self.microstat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": self.stop,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
        }
    
class APIOptions(BaseModel):
    retriever_options: Optional[RetrieverOptions] = None
    generator_options: Optional[GeneratorOptions] = None

class ChatHistory(BaseModel):
    query: str
    response: str

class GeneratorRequest(BaseModel):
    versionName: str
    query: str
    model: str
    history: Optional[list[ChatHistory]] = None
    file_list: Optional[list[FileModel]] = None
    additional_options: Optional[APIOptions] = None
    
class PromptRequest(BaseModel):
    versionName: str
    query: str
    history: Optional[list[ChatHistory]] = None
    file_list: Optional[list[FileModel]] = None
    retriever_options: Optional[RetrieverOptions] = None
    generator_options: Optional[GeneratorOptions] = None