import os
from fastapi import APIRouter, HTTPException
from app.models.schemas import VersionsResponseItem
from config import DOWNLOADS_DIR, SUPPORTED_VERSIONS_FILE

router = APIRouter()

def load_supported_versions():
    try:
        with open(SUPPORTED_VERSIONS_FILE, "r") as f:
            # Read each line, strip whitespace, and ignore empty lines.
            supported_versions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading versions file: {e}")
    return supported_versions

@router.get("/data/versions", response_model=list[VersionsResponseItem])
def list_versions():
    """
    Returns a list of supported version objects.
    The supported versions are read from a file.
    The 'downloaded' flag is determined by checking the downloads directory.
    """
    supported_versions = load_supported_versions()
    downloads_dir = os.path.abspath(DOWNLOADS_DIR)
    versions = []
    for ver in supported_versions:
        file_path = os.path.join(downloads_dir, f"{ver}.csv")
        versions.append({
            "versionName": ver,
            "downloaded": os.path.exists(file_path)
        })
    return versions

@router.get("/data/versions/{version}", response_model=dict)
def get_version_detail(version: str):
    """
    Returns detailed information for a specific version.
    Checks if the version is supported and, if its CSV file is downloaded,
    provides file size and last modified timestamp.
    """
    supported_versions = load_supported_versions()
    if version not in supported_versions:
        raise HTTPException(status_code=404, detail=f"Version {version} is not supported.")
    downloads_dir = os.path.abspath(DOWNLOADS_DIR)
    file_path = os.path.join(downloads_dir, f"{version}.csv")
    detail = {
        "versionName": version,
        "downloaded": os.path.exists(file_path)
    }
    if os.path.exists(file_path):
        stat_info = os.stat(file_path)
        detail.update({
            "file_size": stat_info.st_size,
            "last_modified": stat_info.st_mtime  # UNIX timestamp; convert if needed
        })
    return detail

@router.get("/data/downloaded", response_model=list[str])
def list_downloaded_versions():
    """
    Returns a list of versions that have been downloaded.
    """
    supported_versions = load_supported_versions()
    downloads_dir = os.path.abspath(DOWNLOADS_DIR)
    downloaded = [ver for ver in supported_versions if os.path.exists(os.path.join(downloads_dir, f"{ver}.csv"))]
    return downloaded

@router.get("/data/stats", response_model=dict)
def version_stats():
    """
    Returns statistics about the supported and downloaded versions.
    """
    supported_versions = load_supported_versions()
    downloads_dir = os.path.abspath(DOWNLOADS_DIR)
    downloaded = [ver for ver in supported_versions if os.path.exists(os.path.join(downloads_dir, f"{ver}.csv"))]
    stats = {
        "total_supported": len(supported_versions),
        "total_downloaded": len(downloaded),
        "downloaded_versions": downloaded
    }
    return stats