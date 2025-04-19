from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

from config import DOWNLOADS_DIR

__all__ = ["KaggleDocumentationDownloader"]

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+\.csv$")


class KaggleDocumentationDownloader:
    """Download a single version CSV from the Kaggle dataset and save it locally."""

    def __init__(self, dataset_id: str = "jiyujizai/nextjs-documentation-for-raggin") -> None:
        self.dataset_id = dataset_id
        logging.info("Downloader initialised for dataset '%s'", dataset_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_and_save_version(self, version: str, destination: str | os.PathLike[str] = DOWNLOADS_DIR) -> Path:
        """Fetch `<version>.csv` from Kaggle Hub and write it into *destination*.

        Parameters
        ----------
        version : str
            Semantic version string like ``15.0.0`` or ``v15.0.0``.
        destination : str | PathLike, default ``config.DOWNLOADS_DIR``
            Folder to write the CSV into. Will be created if missing.

        Returns
        -------
        Path
            Absolute path to the saved CSV.

        Raises
        ------
        ValueError
            If *version* is not in ``v<major>.<minor>.<patch>`` format.
        RuntimeError
            If the file cannot be downloaded or saved.
        """

        normalised = self._normalise_version(version)

        logging.info("Loading '%s' from Kaggle dataset '%s'", normalised, self.dataset_id)
        try:
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                self.dataset_id,
                normalised,
            )
        except Exception as exc:  # pragma: no cover – kagglehub failure
            raise RuntimeError(f"Error downloading '{normalised}': {exc}") from exc

        dst_dir = Path(destination).expanduser().resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / normalised

        try:
            df.to_csv(dst_path, index=False)
            logging.info("CSV saved → %s", dst_path)
            return dst_path
        except Exception as exc:  # pragma: no cover – IO error
            raise RuntimeError(f"Error writing csv to '{dst_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_version(raw: str) -> str:
        """Ensure leading 'v' and trailing '.csv', validate format."""
        ver = raw
        if not ver.startswith("v"):
            ver = "v" + ver
        if not ver.endswith(".csv"):
            ver += ".csv"
        if not VERSION_RE.match(ver):
            raise ValueError(
                f"Invalid version '{raw}'. Expected semantic format like 'v15.0.0' or '15.0.0'"
            )
        return ver