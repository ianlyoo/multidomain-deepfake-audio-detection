"""Validate reproducible and redistributable audio-data manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REQUIRED_COLUMNS = (
    "sample_id", "relative_path", "sha256", "source_dataset",
    "source_version", "source_url", "license_id", "license_url",
    "redistribution_allowed", "derivatives_allowed", "attribution",
    "file_fake", "voice_fake", "music_fake", "voice_present",
    "music_present", "generator_family", "generator_model",
    "source_family", "speaker_id", "content_id", "codec",
    "sample_rate", "channels", "duration_seconds", "parent_sha256",
    "transformations_json",
)
LABEL_COLUMNS = (
    "file_fake", "voice_fake", "music_fake",
    "voice_present", "music_present",
)

def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()

def _boolean(value: str, column: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"row {row_number} {column} must be true or false")
    return normalized == "true"

def validate_manifest(path: str | Path, data_root: str | Path | None = None) -> int:
    """Check schema, labels, legal delivery, provenance, and optional hashes."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"manifest columns must be exactly {REQUIRED_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")
    root = Path(data_root) if data_root is not None else None
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        relative_path = row["relative_path"].strip().replace("\\", "/")
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"row {row_number} has an empty or duplicate sample_id")
        if not relative_path or relative_path in seen_paths or ".." in Path(relative_path).parts:
            raise ValueError(f"row {row_number} has an unsafe or duplicate relative_path")
        seen_ids.add(sample_id)
        seen_paths.add(relative_path)
        if not _boolean(row["redistribution_allowed"], "redistribution_allowed", row_number):
            raise ValueError(f"row {row_number} is not deliverable to DACON")
        derivatives_allowed = _boolean(row["derivatives_allowed"], "derivatives_allowed", row_number)
        for required in ("source_dataset", "source_version", "source_url", "license_id", "license_url", "attribution"):
            if not row[required].strip():
                raise ValueError(f"row {row_number} {required} is required")
        known_labels = 0
        for column in LABEL_COLUMNS:
            value = row[column].strip()
            if value:
                if value not in {"0", "1"}:
                    raise ValueError(f"row {row_number} {column} must be empty, 0, or 1")
                known_labels += 1
        if known_labels == 0:
            raise ValueError(f"row {row_number} has no known competition labels")
        transformations = row["transformations_json"].strip() or "[]"
        try:
            decoded = json.loads(transformations)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_number} transformations_json is invalid") from exc
        if not isinstance(decoded, list):
            raise ValueError(f"row {row_number} transformations_json must be a list")
        if decoded and not derivatives_allowed:
            raise ValueError(f"row {row_number} has transformations but forbids derivatives")
        expected_hash = row["sha256"].strip().lower()
        if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
            raise ValueError(f"row {row_number} sha256 is invalid")
        if root is not None:
            audio_path = root / relative_path
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if sha256_file(audio_path) != expected_hash:
                raise ValueError(f"row {row_number} sha256 does not match {audio_path}")
    return len(rows)
