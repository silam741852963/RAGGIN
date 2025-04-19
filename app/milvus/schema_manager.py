from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pymilvus import Collection, CollectionSchema, FieldSchema, connections, utility, DataType

from config import DENSE_VECTOR_DIM, DOWNLOADS_DIR

__all__ = ["MilvusSchemaManager"]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper utils
# -----------------------------------------------------------------------------

def _json_or_empty(val: Any) -> Dict[str, Any]:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return val if isinstance(val, dict) else {}


def _sparse_from_str(val: Any) -> Dict[int, float]:
    if isinstance(val, str):
        try:
            parsed = eval(val)  # noqa: S307 – controlled input: csv we created
            return {int(idx): float(weight) for idx, weight in parsed}
        except Exception:  # pragma: no cover
            return {}
    return val if isinstance(val, dict) else {}


# -----------------------------------------------------------------------------
# Manager
# -----------------------------------------------------------------------------

class MilvusSchemaManager:
    """Create / load a Milvus collection and push CSV rows into it."""

    def __init__(self, collection_name: str, uri: str):
        self.collection_name = collection_name
        self.uri = uri
        self.collection: Collection | None = None
        self._ensure_connection()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _ensure_connection(self) -> None:
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logger.info("Connected to Milvus @ %s", self.uri)

    # ------------------------------------------------------------------
    # Collection + schema
    # ------------------------------------------------------------------

    def _field_schemas(self) -> List[FieldSchema]:
        return [
            FieldSchema("entry_id", DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema("title", DataType.VARCHAR, max_length=255),
            FieldSchema("metadata", DataType.JSON),
            FieldSchema("version", DataType.VARCHAR, max_length=20, is_partition_key=True),
            FieldSchema("text_content", DataType.VARCHAR, max_length=65535),
            FieldSchema("code_content", DataType.VARCHAR, max_length=65535),
            FieldSchema("sparse_title", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("dense_text_content", DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIM),
            FieldSchema("dense_code_snippet", DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIM),
            FieldSchema("tag", DataType.VARCHAR, max_length=255),
        ]

    def create_collection(self) -> Collection:
        self._ensure_connection()
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            logger.info("Dropped existing collection '%s'", self.collection_name)

        schema = CollectionSchema(self._field_schemas(), auto_id=False)
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Strong",
            properties={"partitionkey.isolation": True},
        )
        logger.info("Created collection '%s'", self.collection_name)
        return self.collection

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------

    def create_indices(
        self,
        *,
        m_text: int = 16,
        ef_text: int = 200,
        m_code: int = 16,
        ef_code: int = 200,
    ) -> None:
        """Add SPARSE & HNSW indices and load collection."""
        assert self.collection, "create_collection() first"

        self.collection.create_index("sparse_title", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
        self.collection.create_index(
            "dense_text_content",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": m_text, "efConstruction": ef_text},
            },
        )
        self.collection.create_index(
            "dense_code_snippet",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": m_code, "efConstruction": ef_code},
            },
        )
        self.collection.load()
        logger.info("Collection loaded with indices (text M=%d/ef=%d, code M=%d/ef=%d)", m_text, ef_text, m_code, ef_code)

    # ------------------------------------------------------------------
    # CSV ingest
    # ------------------------------------------------------------------

    def _row_to_entity(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "entry_id": str(row["entry_id"]),
            "title": str(row["title"]),
            "metadata": _json_or_empty(row.get("metadata")),
            "version": str(row.get("version", "")),
            "text_content": str(row.get("text_content", ""))[:65535],
            "code_content": str(row.get("code_content", ""))[:65535],
            "sparse_title": _sparse_from_str(row.get("sparse_title")),
            "dense_text_content": json.loads(row["dense_text_content"]) if isinstance(row.get("dense_text_content"), str) else [0.0] * DENSE_VECTOR_DIM,
            "dense_code_snippet": json.loads(row["dense_code_snippet"]) if isinstance(row.get("dense_code_snippet"), str) else [0.0] * DENSE_VECTOR_DIM,
            "tag": str(row.get("tag", "")),
        }

    def insert_csv(self, csv_path: str | Path) -> int:
        """Insert every row from *csv_path*; returns count inserted."""
        assert self.collection, "create_collection() first"
        df = pd.read_csv(csv_path)
        entities = [self._row_to_entity(row) for _, row in df.iterrows()]
        self.collection.insert(entities)
        logger.info("Inserted %d rows from %s", len(entities), csv_path)
        return len(entities)

    # ------------------------------------------------------------------
    # Full pipeline helper
    # ------------------------------------------------------------------

    def build_from_csv(
        self,
        csv_path: str | Path,
        *,
        m_text: int = 16,
        ef_text: int = 200,
        m_code: int = 16,
        ef_code: int = 200,
    ) -> None:
        self.create_collection()
        self.create_indices(m_text=m_text, ef_text=ef_text, m_code=m_code, ef_code=ef_code)
        self.insert_csv(csv_path)
        self.collection.release()
        logger.info("Collection '%s' ready", self.collection_name)

    # ------------------------------------------------------------------
    # Delete helpers
    # ------------------------------------------------------------------

    def delete_version(self, version: str):
        """Remove every row whose *version* field matches."""
        self._ensure_connection()
        self.collection = Collection(self.collection_name)
        self.collection.load()
        expr = f"version == '{version}'"
        result = self.collection.delete(expr)
        logger.info("Deleted rows for version %s → %s", version, result)