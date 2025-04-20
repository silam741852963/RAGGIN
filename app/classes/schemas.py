from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Pydantic config mixin – snake_case internally / camelCase on the wire
# -----------------------------------------------------------------------------

class _SnakeModel(BaseModel):
    class Config:
        alias_generator = None
        populate_by_name = True
        from_attributes = True


# -----------------------------------------------------------------------------
# Version / dataset models
# -----------------------------------------------------------------------------

class VersionsResponseItem(_SnakeModel):
    version_name: str = Field(...)
    downloaded: bool


class RetrieveRequest(_SnakeModel):
    version_name: str
    m_text: Optional[int] = None
    ef_text: Optional[int] = None
    m_code: Optional[int] = None
    ef_code: Optional[int] = None


# -----------------------------------------------------------------------------
# Collection build request
# -----------------------------------------------------------------------------

class SchemaRequest(_SnakeModel):
    collection_name: str = "nextjs_docs"
    csv_file_path: str
    batch_size: int = 100
    uri: str = "http://standalone:19530"


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

class SearchRequest(_SnakeModel):
    text_query: str
    code_query: str
    version_name: str

    sparse_weight: float = 1.0
    dense_text_weight: float = 1.0
    dense_code_weight: float = 1.0
    top_k: int = 10

    filter_expr: Optional[str] = None
    iterative_filter: bool = False

    radius_sparse: float = 0.5
    range_sparse: float = 0.5
    radius_dense_text: float = 0.5
    range_dense_text: float = 0.5
    radius_dense_code: float = 0.5
    range_dense_code: float = 0.5


# -----------------------------------------------------------------------------
# Retriever / generator option structures
# -----------------------------------------------------------------------------

class RetrieverOptions(_SnakeModel):
    m_text: Optional[int] = None
    ef_text: Optional[int] = None
    m_code: Optional[int] = None
    ef_code: Optional[int] = None

    sparse_weight: float = 1.0
    dense_text_weight: float = 1.0
    dense_code_weight: float = 1.0
    top_k: Optional[int] = None

    filter_expr: Optional[str] = None
    iterative_filter: bool = False

    radius_sparse: float = 0.5
    range_sparse: float = 0.5
    radius_dense_text: float = 0.5
    range_dense_text: float = 0.5
    radius_dense_code: float = 0.5
    range_dense_code: float = 0.5


class GeneratorOptions(_SnakeModel):
    mirostat: Optional[int] = Field(None, alias="microstat")
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: str | list[str] | None = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        if (type(self.stop) is str):
            self.stop = list(self.stop)
        return self.dict(exclude_none=True, by_alias=True)


class APIOptions(_SnakeModel):
    retriever_options: Optional[RetrieverOptions] = None
    generator_options: Optional[GeneratorOptions] = None


# -----------------------------------------------------------------------------
# Generator / prompt requests
# -----------------------------------------------------------------------------

class FileModel(_SnakeModel):
    file_name: str
    file_extension: str
    file_content: str


class ChatHistory(_SnakeModel):
    query: str
    response: str


class GeneratorRequest(_SnakeModel):
    version_name: str
    query: str
    model: str
    history: Optional[List[ChatHistory]] = None
    file_list: Optional[List[FileModel]] = None
    additional_options: Optional[APIOptions] = None


class PromptRequest(_SnakeModel):
    version_name: str
    query: str
    file_list: Optional[List[FileModel]] = None
    retriever_options: Optional[RetrieverOptions] = None
    generator_options: Optional[GeneratorOptions] = None