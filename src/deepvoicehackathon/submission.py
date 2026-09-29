"""Submission CSV and archive validation for DACON 236749."""

from __future__ import annotations

import csv
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .metrics import PREDICTION_COLUMNS

MAX_ZIP_BYTES = 10 * 1024**3
MAX_EXTRACTED_BYTES = 32 * 1024**3
REQUIRED_ROOT_FILES = {"script.py", "requirements.txt"}
ALLOWED_ROOTS = {"model", *REQUIRED_ROOT_FILES}

@dataclass(frozen=True)
class ArchiveReport:
    path: Path
    compressed_bytes: int
    extracted_bytes: int
    file_count: int

def validate_archive(path: str | Path) -> ArchiveReport:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    compressed_bytes = archive_path.stat().st_size
    if compressed_bytes > MAX_ZIP_BYTES:
        raise ValueError("archive exceeds the 10 GiB submission limit")

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos:
            raise ValueError("archive is empty")
        names = {info.filename.rstrip("/") for info in infos}
        roots: set[str] = set()
        extracted_bytes = 0
        file_count = 0
        for info in infos:
            normalized = info.filename.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            if not parts or parts[0] in {"", "."}:
                raise ValueError(f"invalid archive path: {info.filename}")
            if ".." in parts or PurePosixPath(normalized).is_absolute():
                raise ValueError(f"unsafe archive path: {info.filename}")
            roots.add(parts[0])
            if not info.is_dir():
                extracted_bytes += info.file_size
                file_count += 1

        missing = REQUIRED_ROOT_FILES - names
        if missing:
            raise ValueError(f"archive is missing required root files: {sorted(missing)}")
        if not any(name == "model" or name.startswith("model/") for name in names):
            raise ValueError("archive is missing the model/ directory")
        unexpected = roots - ALLOWED_ROOTS
        if unexpected:
            raise ValueError(f"unexpected top-level entries: {sorted(unexpected)}")
        if extracted_bytes > MAX_EXTRACTED_BYTES:
            raise ValueError("archive exceeds the 32 GiB extracted-size limit")
    return ArchiveReport(archive_path, compressed_bytes, extracted_bytes, file_count)

def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)

def validate_prediction_csv(prediction_path: str | Path, sample_path: str | Path) -> int:
    expected_columns = ["ID", *PREDICTION_COLUMNS]
    columns, rows = _read_csv(prediction_path)
    sample_columns, sample_rows = _read_csv(sample_path)
    if columns != expected_columns:
        raise ValueError(f"prediction columns must be exactly {expected_columns}")
    if sample_columns != expected_columns:
        raise ValueError("sample submission has an unexpected schema")
    ids = [row["ID"] for row in rows]
    sample_ids = [row["ID"] for row in sample_rows]
    if ids != sample_ids:
        raise ValueError("prediction IDs/order do not match sample_submission.csv")
    if len(ids) != len(set(ids)):
        raise ValueError("prediction IDs are not unique")
    for row_index, row in enumerate(rows, start=2):
        for column in PREDICTION_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_index} {column} is not numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"row {row_index} {column} must be finite and in [0, 1]")
    return len(rows)
