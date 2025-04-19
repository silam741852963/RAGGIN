from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.downloader.kaggle_downloader import KaggleDocumentationDownloader
from app.milvus.schema_manager import MilvusSchemaManager
from app.classes.schemas import RetrieveRequest
from config import DOWNLOADS_DIR, MILVUS_URI

router = APIRouter(prefix="/version", tags=["version"])

CSV_DIR = Path(DOWNLOADS_DIR).resolve()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _csv_path(version: str) -> Path:
    file = version if version.endswith(".csv") else f"{version}.csv"
    return CSV_DIR / file


def _validate_index_params(m_val: int, ef_val: int, *, name: str) -> None:  # noqa: D401
    """Raise 400 if m/ef are outside allowed bounds."""
    if not 2 <= m_val <= 2048:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: must be between 2 and 2048")
    if ef_val < 1:
        raise HTTPException(status_code=400, detail=f"Invalid ef_{name}: must be at least 1")


@lru_cache(maxsize=1)
def _manager() -> MilvusSchemaManager:
    return MilvusSchemaManager("nextjs_docs", uri=MILVUS_URI)


_downloader = KaggleDocumentationDownloader()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.post("/retrieve")
def retrieve_data(req: RetrieveRequest):
    version = req.version_name
    csv_path = _csv_path(version)
    if csv_path.exists():
        return {"message": f"Version {version} already downloaded", "file_path": str(csv_path)}

    m_text = req.m_text or 16
    ef_text = req.ef_text or 200
    m_code = req.m_code or 16
    ef_code = req.ef_code or 200
    _validate_index_params(m_text, ef_text, name="m_text")
    _validate_index_params(m_code, ef_code, name="m_code")

    try:
        csv_downloaded = _downloader.load_and_save_version(version, destination=CSV_DIR)
        _manager().build_from_csv(
            csv_downloaded,
            m_text=m_text,
            ef_text=ef_text,
            m_code=m_code,
            ef_code=ef_code,
        )
        return {"message": "File retrieved & ingested", "file_path": str(csv_downloaded)}
    except Exception as exc:  # pragma: no cover – generic failure
        logging.exception("retrieve_data failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/delete")
def delete_version(req: RetrieveRequest):
    version = req.version_name
    path = _csv_path(version)
    if not path.exists():
        return {"message": f"Version {version} not downloaded"}

    try:
        _manager().delete_version(version)
        path.unlink()
        logging.info("Deleted CSV %s", path)
        return {"message": f"Version {version} deleted"}
    except Exception as exc:
        logging.exception("Delete version failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/repair")
def repair_version(req: RetrieveRequest):
    version = req.version_name
    path = _csv_path(version)

    m_text = req.m_text or 16
    ef_text = req.ef_text or 200
    m_code = req.m_code or 16
    ef_code = req.ef_code or 200
    _validate_index_params(m_text, ef_text, name="m_text")
    _validate_index_params(m_code, ef_code, name="m_code")

    # delete existing
    if path.exists():
        try:
            _manager().delete_version(version)
            path.unlink()
            logging.info("Removed stale CSV %s", path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error cleaning old data: {exc}") from exc

    # fresh download + ingest
    try:
        new_path = _downloader.load_and_save_version(version, destination=CSV_DIR)
        _manager().build_from_csv(new_path, m_text=m_text, ef_text=ef_text, m_code=m_code, ef_code=ef_code)
        return {"message": f"Version {version} repaired", "file_path": str(new_path)}
    except Exception as exc:
        logging.exception("Repair failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc