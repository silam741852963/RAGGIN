import os, re, logging, json
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config import DOWNLOADS_DIR

class KaggleDocumentationDownloader:
    def __init__(self, dataset_id: str = "jiyujizai/nextjs-documentation-for-raggin"):
        self.dataset_id = dataset_id
        logging.info("Initialized downloader for dataset '%s'.", self.dataset_id)

    def load_and_save_version(self, version: str, destination: str = DOWNLOADS_DIR) -> str:
        # Normalize version string: add "v" prefix and ".csv" suffix if missing.
        if not version.startswith("v"):
            version = "v" + version
        if not version.endswith(".csv"):
            version += ".csv"
        if not re.match(r'^v\d+\.\d+\.\d+\.csv$', version):
            raise ValueError(f"Invalid version format: {version}. Expected format: v<major>.<minor>.<patch>.csv")
        try:
            logging.info("Loading file '%s' from dataset '%s'...", version, self.dataset_id)
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                self.dataset_id,
                version
            )
        except Exception as e:
            raise Exception(f"Error loading file '{version}': {e}")
        os.makedirs(destination, exist_ok=True)
        abs_destination = os.path.abspath(destination)
        file_path = os.path.join(abs_destination, version)
        try:
            df.to_csv(file_path, index=False)   
            logging.info("File saved to: %s", file_path)
            return file_path
        except Exception as e:
            raise Exception(f"Error saving file to '{file_path}': {e}")