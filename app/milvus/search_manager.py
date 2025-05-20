from __future__ import annotations

"""Hybrid vector / lexical retrieval against Milvus with score blending."""

from pathlib import Path
from typing import Any, Dict, List
import heapq
import logging

from FlagEmbedding import BGEM3FlagModel
from pymilvus import Collection, connections

from utils import normalize_distance
from config import MILVUS_URI, MODEL_CACHE_DIR

__all__ = ["SearchManager"]

logger = logging.getLogger(__name__)


class SearchManager:
    """Run sparse, dense‑text & dense‑code searches and merge results."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, collection_name: str, *, uri: str = MILVUS_URI) -> None:
        self.uri = uri
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logger.debug("Connected to Milvus @ %s", self.uri)

        self.collection: Collection = Collection(collection_name)
        self.collection.load()

        self.embedder = BGEM3FlagModel(
            model_name_or_path="BAAI/bge-m3",
            cache_dir=str(Path(MODEL_CACHE_DIR)),
            normalize_embeddings=True,
            return_dense=True,
            return_sparse=True,
            devices=["cpu"],
            use_fp16=False,
        )
        logger.debug("SearchManager ready – collection '%s' loaded", collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> None:
        if not connections.has_connection("default"):
            connections.connect(uri=self.uri)
            logger.debug("Re‑connected to Milvus @ %s", self.uri)

    @staticmethod
    def _filter_expr(version: str, extra: str | None = None) -> str:
        base = f'version == "{version}"'
        return f"{base} && {extra.strip()}" if extra else base

    def _search(
        self,
        field: str,
        query: Any,
        *,
        top_k: int,
        expr: str,
        metric: str,
        radius: float = 0.5,
        range_filter: float = 1,
        extra_params: Dict[str, Any] | None = None,
    ):
        self._ensure_conn()
        params = {"metric_type": metric, "radius": radius, "range": range_filter}
        if extra_params:
            params.update(extra_params)
        return self.collection.search(
            data=[query],
            anns_field=field,
            param=params,
            limit=top_k,
            output_fields=[
                "title",
                "metadata",
                "text_content",
                "code_content",
                "version",
                "tag",
            ],
            expr=expr,
        )[0]

    def _sparse(self, *a, **kw):
        return self._search(*a, metric="IP", **kw)

    def _dense(self, *a, **kw):
        return self._search(*a, metric="COSINE", extra_params={"params": {"nprobe": 10}}, **kw)

    # score helpers -----------------------------------------------------------

    @staticmethod
    def _merge_hits(store: Dict[str, Dict[str, Any]], hits, dist_field: str) -> None:
        for h in hits:
            eid = h.id
            entry = store.setdefault(
                eid,
                {
                    "title": h.entity.get("title"),
                    "metadata": h.entity.get("metadata"),
                    "version": h.entity.get("version"),
                    "text_content": h.entity.get("text_content"),
                    "code_content": h.entity.get("code_content"),
                    "tag": h.entity.get("tag"),
                    "sparse_distance": None,
                    "dense_text_distance": None,
                    "dense_code_distance": None,
                },
            )
            entry[dist_field] = h.distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        text_query: str,
        code_query: str,
        version: str,
        sparse_weight: float = 1.0,
        dense_text_weight: float = 1.0,
        dense_code_weight: float = 1.0,
        top_k: int = 10,
        filter_expr: str | None = None,
        # distance params
        radius_sparse: float = 0.5,
        range_sparse: float = 1,
        radius_dense_text: float = 0.5,
        range_dense_text: float = 1,
        radius_dense_code: float = 0.5,
        range_dense_code: float = 1,
    ) -> List[Dict[str, Any]]:
        if not any([sparse_weight, dense_text_weight, dense_code_weight]):
            raise ValueError("All modality weights are zero – nothing to search.")

        embeds = self.embedder.encode_queries(
            [text_query, code_query], return_dense=True, return_sparse=True
        )
        text_dense, code_dense = embeds["dense_vecs"]
        text_sparse = embeds["lexical_weights"][0]

        expr = self._filter_expr(version, filter_expr)

        sparse_hits = (
            self._sparse("sparse_title", text_sparse, top_k=top_k, expr=expr, radius=radius_sparse, range_filter=range_sparse)
            if sparse_weight
            else []
        )
        dense_text_hits = (
            self._dense("dense_text_content", text_dense, top_k=top_k, expr=expr, radius=radius_dense_text, range_filter=range_dense_text)
            if dense_text_weight
            else []
        )
        dense_code_hits = (
            self._dense("dense_code_snippet", code_dense, top_k=top_k, expr=expr, radius=radius_dense_code, range_filter=range_dense_code)
            if dense_code_weight
            else []
        )

        merged: Dict[str, Dict[str, Any]] = {}
        if sparse_weight:
            self._merge_hits(merged, sparse_hits, "sparse_distance")
        if dense_text_weight:
            self._merge_hits(merged, dense_text_hits, "dense_text_distance")
        if dense_code_weight:
            self._merge_hits(merged, dense_code_hits, "dense_code_distance")

        def _score(e: Dict[str, Any]) -> float:
            score = 0.0
            wsum = 0.0
            if sparse_weight and e["sparse_distance"] is not None:
                score += sparse_weight * normalize_distance(e["sparse_distance"])
                wsum += sparse_weight
            if dense_text_weight and e["dense_text_distance"] is not None:
                score += dense_text_weight * normalize_distance(e["dense_text_distance"])
                wsum += dense_text_weight
            if dense_code_weight and e["dense_code_distance"] is not None:
                score += dense_code_weight * normalize_distance(e["dense_code_distance"])
                wsum += dense_code_weight
            return score / wsum if wsum else 0.0

        # top‑k
        best = heapq.nlargest(top_k, merged.values(), key=_score)
        for itm in best:
            itm["combined_score"] = _score(itm)
        return best