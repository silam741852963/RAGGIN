import logging
import math
import os
from pymilvus import Collection, connections
from scipy.sparse import csr_matrix
import torch
from FlagEmbedding import BGEM3FlagModel
from util import normalize_distance
from config import MILVUS_URI, MODEL_CACHE_DIR

class HybridSearchManager:
    def __init__(self, collection_name, uri=MILVUS_URI):
        self.uri = uri
        # Ensure a connection is established upon initialization.
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logging.debug(f"Connected to Milvus at {self.uri} during initialization.")
        else:
            logging.debug(f"Using existing Milvus connection at {self.uri}.")
        self.collection = Collection(collection_name)
        self.collection.load()
    
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
    
        # Initialize BGEM3FlagModel using the model from cache.
        self.embedding_function = BGEM3FlagModel(
            model_name_or_path="BAAI/bge-m3",
            normalize_embeddings=True,
            use_fp16=False,
            devices=[device],
            cache_dir=MODEL_CACHE_DIR,
            return_dense=True,
            return_sparse=True,  # Ensures we can retrieve both dense & sparse embeddings
        )
        logging.debug(f"HybridSearchManager initialized with BGEM3FlagModel on device: {device}")

    def _ensure_connection(self):
        # Ensure a default connection exists before performing DB tasks.
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logging.debug(f"Re-connected to Milvus at {self.uri} in _ensure_connection.")
        else:
            logging.debug("Milvus connection is active.")

    def _get_filter_expr(self, filter_expr, versionName):
        """
        Combine the partition filter based on versionName with any additional filter expression.
        Returns a string of the form:
            version == "versionName" && <filter_expr>
        If filter_expr is empty, returns only the partition filter.
        """
        partition_filter = f'version == "{versionName}"'
        if filter_expr and filter_expr.strip():
            combined = f'{partition_filter} && {filter_expr.strip()}'
            logging.debug(f"Combined filter expression: {combined}")
            return combined
        logging.debug(f"Using partition filter only: {partition_filter}")
        return partition_filter

    def _search_field(self, field_name, query_data, limit, filter_expr, radius, range_filter, metric_type, extra_params=None, iterative_filter=False):
        # Ensure connection is active before searching.
        self._ensure_connection()
        params = {"metric_type": metric_type, "radius": radius, "range": range_filter}
        if extra_params:
            params.update(extra_params)
        if iterative_filter:
            params["hints"] = "iterative_filter"
        valid_filter = filter_expr
        logging.debug(f"Performing search on field '{field_name}' with params: {params} and filter: {valid_filter}")
        results = self.collection.search(
            data=[query_data],
            anns_field=field_name,
            param=params,
            limit=limit,
            output_fields=["title", "metadata", "text_content", "code_content", "version", "tag"],
            expr=valid_filter
        )[0]
        logging.debug(f"Search on '{field_name}' returned {len(results)} result set(s).")
        return results

    def sparse_search(self, query_vector, limit, filter_expr, radius_sparse, range_sparse, iterative_filter=False):
        return self._search_field("sparse_title", query_vector, limit, filter_expr, radius_sparse, range_sparse, "IP", iterative_filter=iterative_filter)

    def dense_text_search(self, query_dense_text_embedding, limit, filter_expr, radius_dense_text, range_dense_text, iterative_filter=False):
        return self._search_field(
            "dense_text_content",
            query_dense_text_embedding,
            limit,
            filter_expr,
            radius_dense_text,
            range_dense_text,
            "COSINE",
            extra_params={"params": {"nprobe": 10}},
            iterative_filter=iterative_filter,
        )

    def code_search(self, query_dense_code_embedding, limit, filter_expr, radius_dense_code, range_dense_code, iterative_filter=False):
        return self._search_field(
            "dense_code_snippet",
            query_dense_code_embedding,
            limit,
            filter_expr,
            radius_dense_code,
            range_dense_code,
            "COSINE",
            extra_params={"params": {"nprobe": 10}},
            iterative_filter=iterative_filter,
        )
    
    def update_results(self, combined_results_dict, hits, distance_field, version_name):
        """
        Updates the combined results dictionary using the given hits list for the specified distance field.
        
        Args:
            combined_results_dict (dict): The dictionary to update.
            hits (list): A list of hit dictionaries.
            distance_field (str): Field name to update (e.g., "sparse_distance", "dense_text_distance", or "dense_code_distance").
            version_name (str): Default version name if not provided in the hit.
        """
        for hit in hits:
            entry_id = hit.id
            distance = hit.distance
            entity = hit.entity

            if entry_id not in combined_results_dict:
                combined_results_dict[entry_id] = {
                    "title": entity.get("title"),
                    "metadata": entity.get("metadata"),
                    "version": entity.get("version"),
                    "text_content": entity.get("text_content"),
                    "code_content": entity.get("code_content"),
                    "tag": entity.get("tag"),
                    "sparse_distance": None,
                    "dense_text_distance": None,
                    "dense_code_distance": None,
                }
            combined_results_dict[entry_id][distance_field] = distance

    def hybrid_search(
        self,
        text_query: str,
        code_query: str,
        versionName,
        sparseWeight: float = 1.0,
        denseTextWeight: float = 1.0,
        denseCodeWeight: float = 1.0,
        topK: int = 10,
        filter_expr: str = "",
        iterativeFilter: bool = False,
        radius_sparse: float = 0.5,
        range_sparse: float = 0.5,
        radius_dense_text: float = 0.5,
        range_dense_text: float = 0.5,
        radius_dense_code: float = 0.5,
        range_dense_code: float = 0.5,
    ):
        if sparseWeight == 0 and denseTextWeight == 0 and denseCodeWeight == 0:
            raise ValueError("At least one search modality must have a non-zero weight.")
        
        # Construct filter using versionName as partition key.
        combined_filter = self._get_filter_expr(filter_expr, versionName)
        logging.debug(f"Using combined filter: {combined_filter}")
        
        # Encode queries
        logging.debug("Encoding text query...")
        result = self.embedding_function.encode_queries(
            queries=[text_query],
            return_dense=True,
            return_sparse=True
        )
        text_dense_vec = result["dense_vecs"][0]
        text_sparse_vec = result["lexical_weights"][0]
        logging.debug("Text query embeddings obtained.")
        logging.debug(text_dense_vec)
        logging.debug(text_sparse_vec)
        
        logging.debug("Encoding code query...")
        result = self.embedding_function.encode_queries(
            queries=[code_query],
            return_dense=True,
            return_sparse=True
        )
        code_dense_vec = result["dense_vecs"][0]
        logging.debug("Code query embeddings obtained.")
        
        # Log embedding shapes if available.
        try:
            logging.debug(f"Text dense vector shape: {text_dense_vec.shape}")
            logging.debug(f"Text sparse vector: {text_sparse_vec}")
            logging.debug(f"Code dense vector shape: {code_dense_vec.shape}")
        except Exception as e:
            logging.warning("Could not log embedding shapes: %s", e)

        sparse_results = []
        dense_text_results = []
        code_results = []
        
        if sparseWeight != 0:
            try:
                logging.debug("Performing sparse search...")
                sparse_results = self.sparse_search(text_sparse_vec, topK, combined_filter, radius_sparse, range_sparse, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in sparse search: %s", e)
        if denseTextWeight != 0:
            try:
                logging.debug("Performing dense text search...")
                dense_text_results = self.dense_text_search(text_dense_vec, topK, combined_filter, radius_dense_text, range_dense_text, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in dense text search: %s", e)
        if denseCodeWeight != 0:
            try:
                logging.debug("Performing dense code search...")
                code_results = self.code_search(code_dense_vec, topK, combined_filter, radius_dense_code, range_dense_code, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in dense code search: %s", e)

        combined_results_dict = {}

        # Process each modality one at a time.
        if sparseWeight != 0:
            self.update_results(combined_results_dict, sparse_results, "sparse_distance", versionName)
        if denseTextWeight != 0:
            self.update_results(combined_results_dict, dense_text_results, "dense_text_distance", versionName)
        if denseCodeWeight != 0:
            self.update_results(combined_results_dict, code_results, "dense_code_distance", versionName)

        combined_results = []
        for entry_id, data in combined_results_dict.items():
            score = 0
            weight_sum = 0

            if sparseWeight and data.get("sparse_distance") is not None:
                raw = data["sparse_distance"]
                norm = normalize_distance(raw)
                score += sparseWeight * norm
                weight_sum += sparseWeight

            if denseTextWeight and data.get("dense_text_distance") is not None:
                raw = data["dense_text_distance"]
                norm = normalize_distance(raw)
                score += denseTextWeight * norm
                weight_sum += denseTextWeight

            if denseCodeWeight and data.get("dense_code_distance") is not None:
                raw = data["dense_code_distance"]
                norm = normalize_distance(raw)
                score += denseCodeWeight * norm
                weight_sum += denseCodeWeight

            combined_score = score / weight_sum if weight_sum else 0
            data["combined_score"] = combined_score
            combined_results.append(data)

        # Sort the results by combined score in descending order.
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results