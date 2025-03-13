import logging
from pymilvus import Collection, MilvusClient
from scipy.sparse import csr_matrix
import torch
from config import DENSE_VECTOR_DIM, MILVUS_URI

class HybridSearchManager:
    def __init__(self, collection_name):
        # self.collection = Collection(collection_name)
        self.client = MilvusClient(uri=MILVUS_URI)
        self.collection_name = collection_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Replace dummy_embedding_function with your real embedding logic.
        self.embedding_function = self.dummy_embedding_function
        logging.info("HybridSearchManager initialized on device: %s", device)

    def dummy_embedding_function(self, queries):
        return {
            "sparse": [(0, 0.5)],  # Dummy sparse embedding
            "dense": [[0.1] * DENSE_VECTOR_DIM for _ in queries]
        }

    def sparse_search(self, query_vector, limit, filter_expr, radius, range_filter):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="sparse_title",
            search_params={"metric_type": "IP", "radius": radius, "range": range_filter},
            limit=limit,
            output_fields=["entry_id", "title", "metadata", "text_content", "code_content", "version", "tag"],
            # expr=filter_expr
        )
        return results

    def dense_text_search(self, query_dense_text_embedding, limit, filter_expr, radius, range_filter):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_dense_text_embedding],
            anns_field="dense_text_content",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}, "radius": radius, "range": range_filter},
            limit=limit,
            output_fields=["entry_id", "title", "metadata", "text_content", "code_content", "version", "tag"],
            # expr=filter_expr
        )
        return results

    def code_search(self, query_dense_code_embedding, limit, filter_expr, radius, range_filter):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_dense_code_embedding],
            anns_field="dense_code_snippet",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}, "radius": radius, "range": range_filter},
            limit=limit,
            output_fields=["entry_id", "title", "metadata", "text_content", "code_content", "version", "tag"],
            # expr=filter_expr``
        )
        return results

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

    def hybrid_search(self, versionName, sparseWeight, denseTextWeight, denseCodeWeight, topK, filter_expr, iterativeFilter, radius, range_filter):
        # Dummy queries for embedding extraction.
        text_query = "dummy text query"
        code_query = "dummy code query"
        text_query_embeddings = self.embedding_function([text_query])
        query_sparse_embedding = csr_matrix(text_query_embeddings.get("sparse", None))
        query_dense_text_embedding = text_query_embeddings.get("dense", None)[0]
        code_query_embeddings = self.embedding_function([code_query])
        query_dense_code_embedding = code_query_embeddings.get("dense", None)[0]

        # Perform searches with the provided filter, radius, and range_filter.
        sparse_results = self.sparse_search(query_sparse_embedding, topK, filter_expr, radius, range_filter)
        dense_text_results = self.dense_text_search(query_dense_text_embedding, topK, filter_expr, radius, range_filter)
        code_results = self.code_search(query_dense_code_embedding, topK, filter_expr, radius, range_filter)

        combined_results_dict = {}
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
                        "dense_text_distance": 0,
                        "dense_code_distance": 0
                    }
                else:
                    combined_results_dict[entry_id]["sparse_distance"] = distance

        for hits in dense_text_results:
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
                            "sparse_distance": 0,
                            "dense_text_distance": distance,
                            "dense_code_distance": 0
                        }
                    else:
                        combined_results_dict[entry_id]["dense_text_distance"] = distance
                except Exception as e:
                    logging.exception("Error processing dense text hit:")

        for hits in code_results:
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
                            "sparse_distance": 0,
                            "dense_text_distance": 0,
                            "dense_code_distance": distance
                        }
                    else:
                        combined_results_dict[entry_id]["dense_code_distance"] = distance
                except Exception as e:
                    logging.exception("Error processing dense code hit:")

        sum_weights = sparseWeight + denseTextWeight + denseCodeWeight or 1
        combined_results = []
        for entry_id, data in combined_results_dict.items():
            combined_score = (data["sparse_distance"] * sparseWeight +
                              data["dense_text_distance"] * denseTextWeight +
                              data["dense_code_distance"] * denseCodeWeight) / sum_weights
            data["combined_score"] = combined_score
            combined_results.append(data)
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:topK]