import logging
import math
from pymilvus import Collection, connections
from scipy.sparse import csr_matrix
import torch
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from util import normalize_distance
from config import DENSE_VECTOR_DIM, MILVUS_URI

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use the actual BGEM3EmbeddingFunction for embeddings.
        self.embedding_function = BGEM3EmbeddingFunction(model_name='BAAI/bge-m3', use_fp16=False, device=device)
        logging.debug(f"HybridSearchManager initialized on device: {device}")

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
            partition_key == "versionName" && <filter_expr>
        If filter_expr is empty, returns only the partition filter.
        """
        partition_filter = f'partition_key == "{versionName}"'
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
        valid_filter = filter_expr  # Already prepared by caller.
        logging.debug(f"Performing search on field '{field_name}' with params: {params} and filter: {valid_filter}")
        return self.collection.search(
            data=[query_data],
            anns_field=field_name,
            param=params,
            limit=limit,
            output_fields=["entry_id", "title", "metadata", "text_content", "code_content", "version", "tag"],
            expr=valid_filter
        )

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

    def process_hit(self, hit):
        if isinstance(hit, dict):
            entity = hit.get("entity", {})
            distance = hit.get("distance", 0)
        else:
            entity = getattr(hit, "entity", {})
            distance = getattr(hit, "distance", 0)
        if isinstance(entity, list):
            entity = entity[0] if entity else {}
        if isinstance(entity, dict) and "entry_id" not in entity:
            entity["entry_id"] = hit.get("id", "N/A") if isinstance(hit, dict) else "N/A"
        return entity, distance

    def _update_dense_results(self, combined_results_dict, results, versionName, distance_key):
        for hits in results:
            for hit in hits:
                try:
                    if isinstance(hit, dict):
                        data = hit.get("entity", hit)
                        entry_id = data.get("entry_id", "N/A")
                        distance = hit.get("distance", 0)
                    else:
                        entry_id = getattr(hit, "entry_id", "N/A")
                        distance = getattr(hit, "distance", 0)
                    if entry_id not in combined_results_dict:
                        combined_results_dict[entry_id] = {
                            "title": data.get("title", "N/A"),
                            "metadata": data.get("metadata", {}),
                            "version": data.get("version", versionName),
                            "textContent": data.get("text_content", ""),
                            "codeContent": data.get("code_content", ""),
                            "tag": data.get("tag", ""),
                            "sparse_distance": None,
                            "dense_text_distance": None,
                            "dense_code_distance": None,
                        }
                    combined_results_dict[entry_id][distance_key] = distance
                except Exception as e:
                    logging.exception("Error processing dense hit for %s", distance_key)

    def hybrid_search(
        self,
        text_query: str,      # Required: no default
        code_query: str,      # Required: no default
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
        
        # Obtain embeddings for the queries.
        text_query_embeddings = self.embedding_function([text_query])
        query_sparse_embedding = csr_matrix(text_query_embeddings.get("sparse", None))
        query_dense_text_embedding = text_query_embeddings.get("dense", None)[0]
        code_query_embeddings = self.embedding_function([code_query])
        query_dense_code_embedding = code_query_embeddings.get("dense", None)[0]

        # Perform searches only if corresponding weight is non-zero.
        sparse_results = []
        dense_text_results = []
        code_results = []
        
        if sparseWeight != 0:
            try:
                logging.debug("Performing sparse search...")
                sparse_results = self.sparse_search(query_sparse_embedding, topK, combined_filter, radius_sparse, range_sparse, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in sparse search: %s", e)
        if denseTextWeight != 0:
            try:
                logging.debug("Performing dense text search...")
                dense_text_results = self.dense_text_search(query_dense_text_embedding, topK, combined_filter, radius_dense_text, range_dense_text, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in dense text search: %s", e)
        if denseCodeWeight != 0:
            try:
                logging.debug("Performing dense code search...")
                code_results = self.code_search(query_dense_code_embedding, topK, combined_filter, radius_dense_code, range_dense_code, iterative_filter=iterativeFilter)
            except Exception as e:
                logging.exception("Error in code search: %s", e)

        combined_results_dict = {}
        # Process sparse results.
        if sparseWeight != 0:
            for hits in sparse_results:
                for hit in hits:
                    entity, distance = self.process_hit(hit)
                    entry_id = entity.get("entry_id", "N/A")
                    if entry_id not in combined_results_dict:
                        combined_results_dict[entry_id] = {
                            "title": entity.get("title", "N/A"),
                            "metadata": entity.get("metadata", {}),
                            "version": entity.get("version", versionName),
                            "textContent": entity.get("text_content", ""),
                            "codeContent": entity.get("code_content", ""),
                            "tag": entity.get("tag", ""),
                            "sparse_distance": distance,
                            "dense_text_distance": None,
                            "dense_code_distance": None,
                        }
                    else:
                        combined_results_dict[entry_id]["sparse_distance"] = distance

        if denseTextWeight != 0:
            self._update_dense_results(combined_results_dict, dense_text_results, versionName, "dense_text_distance")
        if denseCodeWeight != 0:
            self._update_dense_results(combined_results_dict, code_results, versionName, "dense_code_distance")

        # Compute combined score using weighted average over only modalities that were queried.
        combined_results = []
        for entry_id, data in combined_results_dict.items():
            score = 0
            weight_sum = 0
            if sparseWeight != 0 and data.get("sparse_distance") is not None:
                norm = normalize_distance(data["sparse_distance"])
                score += sparseWeight * norm
                weight_sum += sparseWeight
            if denseTextWeight != 0 and data.get("dense_text_distance") is not None:
                norm = normalize_distance(data["dense_text_distance"])
                score += denseTextWeight * norm
                weight_sum += denseTextWeight
            if denseCodeWeight != 0 and data.get("dense_code_distance") is not None:
                norm = normalize_distance(data["dense_code_distance"])
                score += denseCodeWeight * norm
                weight_sum += denseCodeWeight
            combined_score = score / weight_sum if weight_sum else 0
            data["combined_score"] = combined_score
            combined_results.append(data)

        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        logging.debug(f"Hybrid search completed. Returning top {topK} results.")
        return combined_results[:topK]
