from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter, HTTPException

from app.classes.schemas import VersionsResponseItem
from config import DOWNLOADS_DIR, SUPPORTED_VERSIONS_FILE

router = APIRouter(prefix="/data", tags=["data"])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_supported_versions() -> List[str]:
    """Read the list of supported Next.js versions from the configured file."""
    try:
        with open(SUPPORTED_VERSIONS_FILE, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    except Exception as exc:  # pragma: no cover – bubbled as HTTP error
        raise HTTPException(status_code=500, detail=f"Error reading versions file: {exc}") from exc


def _csv_path(version: str) -> Path:
    """Return the absolute path to <downloads_dir>/<version>.csv"""
    return Path(DOWNLOADS_DIR).resolve() / f"{version}.csv"


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.get("/versions", response_model=List[VersionsResponseItem])
def list_versions() -> List[VersionsResponseItem]:
    """Return all supported versions and a *downloaded* flag for each."""
    supported = load_supported_versions()
    return [
        VersionsResponseItem(
            version_name=ver,
            downloaded=_csv_path(ver).exists(),
        )
        for ver in supported
    ]


@router.get("/versions/{version}", response_model=Dict[str, str | bool | int | float])
def get_version_detail(version: str):
    """Return metadata (size, mtime) for one version, 404 if unsupported."""
    if version not in load_supported_versions():
        raise HTTPException(status_code=404, detail=f"Version '{version}' is not supported")

    path = _csv_path(version)
    detail: Dict[str, str | bool | int | float] = {
        "version_name": version,
        "downloaded": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        detail.update(file_size=stat.st_size, last_modified=stat.st_mtime)
    return detail


@router.get("/downloaded", response_model=List[str])
def list_downloaded_versions() -> List[str]:
    """Return only versions whose CSV file exists in the downloads directory."""
    return [ver for ver in load_supported_versions() if _csv_path(ver).exists()]


@router.get("/stats", response_model=Dict[str, int | List[str]])
def version_stats():
    """Return simple counts & list of downloaded versions."""
    supported = load_supported_versions()
    downloaded = [ver for ver in supported if _csv_path(ver).exists()]
    return {
        "total_supported": len(supported),
        "total_downloaded": len(downloaded),
        "downloaded_versions": downloaded,
    }