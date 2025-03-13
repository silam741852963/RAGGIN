import json, logging
import pandas as pd
from pymilvus import (
    connections, Collection, utility,
    DataType, FieldSchema, CollectionSchema
)
from config import DENSE_VECTOR_DIM
import numpy as np

class MilvusSchemaManager:
    def __init__(self, collection_name, uri):
        self.collection_name = collection_name
        self.uri = uri
        self.collection = None

    def _ensure_connection(self):
        # Check if the default connection exists; if not, create it.
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logging.info("Connected to Milvus at %s", self.uri)

    def create_field_schemas(self):
        fields = [
            FieldSchema(
                name="entry_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=255,
                description="Unique identifier for the entry."
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Title of the entry."
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Custom metadata for the entry."
            ),
            FieldSchema(
                name="version",
                dtype=DataType.VARCHAR,
                max_length=20,
                description="Framework version associated with the entry.",
                is_partition_key=True,
            ),
            FieldSchema(
                name="text_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Raw text content of the entry."
            ),
            FieldSchema(
                name="code_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Code snippet content of the entry."
            ),
            FieldSchema(
                name="sparse_title",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="Sparse vector embedding for the title."
            ),
            FieldSchema(
                name="dense_text_content",
                dtype=DataType.FLOAT_VECTOR,
                dim=DENSE_VECTOR_DIM,
                description="Dense vector embedding for the text content."
            ),
            FieldSchema(
                name="dense_code_snippet",
                dtype=DataType.FLOAT_VECTOR,
                dim=DENSE_VECTOR_DIM,
                description="Dense vector embedding for the code content aggregated by average pooling."
            ),
            FieldSchema(
                name="tag",
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Tag for the entry."
            ),
        ]
        return fields

    def create_collection(self):
        self._ensure_connection()
        schema = CollectionSchema(
            fields=self.create_field_schemas(),
            description="Schema for documentation entries, sections, and aggregated code snippet embeddings.",
            auto_id=False,
        )
        if utility.has_collection(self.collection_name):
            coll = Collection(self.collection_name)
            coll.drop()
            logging.info("Collection '%s' dropped.", self.collection_name)
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Strong",
            properties={"partitionkey.isolation": True}
        )
        logging.info("Collection '%s' created successfully.", self.collection_name)
        return self.collection

    def create_indices(self, m_text: int = 16, ef_text: int = 200, m_code: int = 16, ef_code: int = 200):
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index("sparse_title", sparse_index)
        
        dense_text_index = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": m_text, "efConstruction": ef_text},
        }
        self.collection.create_index("dense_text_content", dense_text_index)
        
        dense_code_index = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": m_code, "efConstruction": ef_code},
        }
        self.collection.create_index("dense_code_snippet", dense_code_index)
        
        self.collection.load()
        logging.info(
            "Collection loaded with text index (M=%d, efConstruction=%d) and code index (M=%d, efConstruction=%d).",
            m_text, ef_text, m_code, ef_code,
        )

    def safe_parse(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

    def parse_sparse_vector(self, value):
        if isinstance(value, str):
            try:
                sparse_data = eval(value)
                return {int(item[0]): float(item[1]) for item in sparse_data}
            except (SyntaxError, ValueError, TypeError):
                return {}   
        return value

    def enforce_types(self, row):
        entry_id = str(row["entry_id"])
        title = str(row["title"])
        metadata = self.safe_parse(row["metadata"]) if pd.notna(row["metadata"]) else {}
        version = str(row["version"]) if pd.notna(row["version"]) else ""
        text_content = str(row["text_content"])[:65535] if pd.notna(row["text_content"]) else ""
        code_content = str(row["code_content"]) if pd.notna(row["code_content"]) else ""
        sparse_title = self.parse_sparse_vector(row["sparse_title"])
        dense_text_content = (
            json.loads(row["dense_text_content"])
            if pd.notna(row["dense_text_content"]) and isinstance(row["dense_text_content"], str)
            else [0.0] * DENSE_VECTOR_DIM
        )
        dense_code_snippet = (
            json.loads(row["dense_code_snippet"])
            if pd.notna(row["dense_code_snippet"]) and isinstance(row["dense_code_snippet"], str)
            else [0.0] * DENSE_VECTOR_DIM
        )
        tag = str(row["tag"]) if pd.notna(row["tag"]) else ""
        return {
            "entry_id": entry_id,
            "title": title,
            "metadata": metadata,
            "version": version,
            "text_content": text_content,
            "code_content": code_content,
            "sparse_title": sparse_title,
            "dense_text_content": dense_text_content,
            "dense_code_snippet": dense_code_snippet,
            "tag": tag,
        }

    def insert_data(self, file_path):
        """Reads the entire CSV file and inserts all rows into Milvus at once."""
        self._ensure_connection()
        data = pd.read_csv(file_path)
        entities = [self.enforce_types(row) for _, row in data.iterrows()]
        self.collection.insert(entities)
        logging.info("Inserted %d records from file %s.", len(entities), file_path)

    def run(self, csv_file_path, m_text: int = 16, ef_text: int = 200, m_code: int = 16, ef_code: int = 200):
        self._ensure_connection()
        self.create_collection()
        self.create_indices(m_text, ef_text, m_code, ef_code)
        self.insert_data(csv_file_path)
        self.collection.release()
        logging.info("Collection '%s' is ready.", self.collection_name)
        
    def delete_version(self, version: str):
        """
        Deletes all documents in the collection that match the provided version using a delete expression.
        """
        self._ensure_connection()
        # Re-instantiate and load the collection.
        self.collection = Collection(self.collection_name)
        self.collection.load()
        expr = f"version == '{version}'"
        result = self.collection.delete(expr)
        logging.info("Deleted documents for version %s using expression '%s': %s", version, expr, result)
        return result