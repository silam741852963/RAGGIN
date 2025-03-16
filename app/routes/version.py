import os
import logging
from fastapi import APIRouter, HTTPException
from app.downloader.kaggle_downloader import KaggleDocumentationDownloader
from app.milvus.schema_manager import MilvusSchemaManager
from app.classes.schemas import RetrieveRequest
from config import DOWNLOADS_DIR, MILVUS_URI

router = APIRouter()

def get_csv_file_path(version: str) -> str:
    version_file = version if version.endswith(".csv") else f"{version}.csv"
    return os.path.join(DOWNLOADS_DIR, version_file)

@router.post("/retrieve")
def retrieve_data(request: RetrieveRequest):
    """
    Downloads a CSV file from Kaggle for the given versionName (if not already downloaded)
    and then inserts the data into Milvus using the provided index parameters.
    """
    version = request.versionName
    csv_file = get_csv_file_path(version)
    if os.path.exists(csv_file):
        return {"message": f"Version {version} is already downloaded.", "file_path": csv_file}
    
    # Retrieve index parameters for text and code
    m_text = request.m_text if request.m_text is not None else 16
    ef_text = request.ef_text if request.ef_text is not None else 200
    m_code = request.m_code if request.m_code is not None else 16
    ef_code = request.ef_code if request.ef_code is not None else 200

    # Validate parameters:
    if not (2 <= m_text <= 2048):
        raise HTTPException(status_code=400, detail="Invalid m_text: must be between 2 and 2048.")
    if not (2 <= m_code <= 2048):
        raise HTTPException(status_code=400, detail="Invalid m_code: must be between 2 and 2048.")
    if ef_text < 1:
        raise HTTPException(status_code=400, detail="Invalid ef_text: must be at least 1.")
    if ef_code < 1:
        raise HTTPException(status_code=400, detail="Invalid ef_code: must be at least 1.")
    
    try:
        downloader = KaggleDocumentationDownloader()
        file_path = downloader.load_and_save_version(version, destination=DOWNLOADS_DIR)
        manager = MilvusSchemaManager("nextjs_docs", uri=MILVUS_URI)
        # Updated run method now accepts separate parameters for text and code indices.
        manager.run(file_path, m_text, ef_text, m_code, ef_code)
        return {"message": "File retrieved and data inserted successfully", "file_path": file_path}
    except Exception as e:
        logging.exception("Error in retrieve_data endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete")
def delete_version(request: RetrieveRequest):
    """
    Deletes a version: checks if the version CSV is downloaded,
    removes its data from Milvus, and deletes the CSV file.
    """
    version = request.versionName
    csv_file = get_csv_file_path(version)
    if not os.path.exists(csv_file):
        return {"message": f"Version {version} is not downloaded."}
    try:
        manager = MilvusSchemaManager("nextjs_docs", uri=MILVUS_URI)
        manager.delete_version(version)
    except Exception as e:
        logging.exception("Error deleting version from DB")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        os.remove(csv_file)
        logging.info("Deleted CSV file for version %s", version)
    except Exception as e:
        logging.exception("Error deleting CSV file")
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Version {version} deleted successfully."}

@router.post("/repair")
def repair_version(request: RetrieveRequest):
    """
    Repairs a version by deleting its data from Milvus (if present) and
    re-downloading and re-inserting the data using the provided index parameters.
    """
    version = request.versionName
    csv_file = get_csv_file_path(version)
    
    m_text = request.m_text if request.m_text is not None else 16
    ef_text = request.ef_text if request.ef_text is not None else 200
    m_code = request.m_code if request.m_code is not None else 16
    ef_code = request.ef_code if request.ef_code is not None else 200

    if not (2 <= m_text <= 2048):
        raise HTTPException(status_code=400, detail="Invalid m_text: must be between 2 and 2048.")
    if not (2 <= m_code <= 2048):
        raise HTTPException(status_code=400, detail="Invalid m_code: must be between 2 and 2048.")
    if ef_text < 1:
        raise HTTPException(status_code=400, detail="Invalid ef_text: must be at least 1.")
    if ef_code < 1:
        raise HTTPException(status_code=400, detail="Invalid ef_code: must be at least 1.")
    
    # Delete existing data and CSV if present
    if os.path.exists(csv_file):
        try:
            manager = MilvusSchemaManager("nextjs_docs", uri=MILVUS_URI)
            manager.delete_version(version)
            os.remove(csv_file)
            logging.info("Deleted CSV file for version %s during repair", version)
        except Exception as e:
            logging.exception("Error deleting version during repair")
            raise HTTPException(status_code=500, detail=str(e))
    # Re-run retrieval process with index parameters
    try:
        downloader = KaggleDocumentationDownloader()
        new_file_path = downloader.load_and_save_version(version, destination=DOWNLOADS_DIR)
        manager = MilvusSchemaManager("nextjs_docs", uri=MILVUS_URI)
        manager.run(new_file_path, m_text, ef_text, m_code, ef_code)
        return {"message": f"Version {version} repaired successfully.", "file_path": new_file_path}
    except Exception as e:
        logging.exception("Error repairing version")
        raise HTTPException(status_code=500, detail=str(e))