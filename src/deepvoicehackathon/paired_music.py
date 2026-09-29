"""Deterministic, license-aware pairing of FMA bona fide music with Echoes fakes.

Backs tools/extract_echoes_fma_subset.py. Standard library only, so selection,
path safety, and manifest logic stay testable without pandas, librosa, or the
multi-gigabyte source archives.

Design notes
------------
*   Only exact unique matches from the data-audit CSV are eligible, filtered to
    one FMA subset (small by default). At most 119 originals are ever unpacked,
    never the full 8,000 track archive.
*   FMA member paths are derived arithmetically from the track id
    (fma_small/<first-three-digits>/<six-digit-id>.mp3), so planning needs no
    directory listing and therefore no complete archive.
*   Every archive member name and every destination path is validated before any
    write, blocking zip-slip and absolute-path escapes.
*   Audio bytes are copied verbatim; hashes describe exactly what is on disk.
*   The canonical CSV matches deepvoicehackathon.manifest.REQUIRED_COLUMNS
    exactly so the existing validator accepts it unchanged. A JSONL sidecar
    carries richer provenance (content_group, transformation_allowed, licence
    detail, pairing lineage) that the fixed CSV schema has no column for.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from deepvoicehackathon.manifest import REQUIRED_COLUMNS, sha256_file

FMA_SOURCE_URL = "https://github.com/mdeff/fma"
ECHOES_LICENSE_ID = "CC-BY-SA-4.0"
ECHOES_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
ECHOES_SOURCE_URL = "https://arxiv.org/abs/2603.23667"
FMA_AUDIO_PREFIX = "fma_small"
ECHOES_MEMBER_PREFIX = "Echoes"

# Labels known with certainty for these corpora. Voice labels stay empty on
# purpose: neither dataset documents voice presence per file, and the manifest
# validator accepts partial DACON labels.
REAL_LABELS = {"file_fake": "0", "music_fake": "0", "music_present": "1"}
FAKE_LABELS = {"file_fake": "1", "music_fake": "1", "music_present": "1"}

_BITRATES_V1_L3 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_BITRATES_V2_L3 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


class SubsetExtractionError(RuntimeError):
    """Base class for actionable, self-service extraction failures."""


class ArchiveIncompleteError(SubsetExtractionError):
    """An archive is absent, truncated, or still downloading."""


class UnsafeMemberError(SubsetExtractionError):
    """An archive member or destination path escaped its root."""


class MatchSelectionError(SubsetExtractionError):
    """The match audit could not be read or produced no eligible rows."""


@dataclass(frozen=True)
class LicenseTerms:
    """Normalized licence facts derived from FMA metadata."""

    license_id: str
    license_url: str
    redistribution_allowed: bool
    derivatives_allowed: bool
    commercial_use_allowed: bool
    raw_title: str

    @property
    def transformation_allowed(self) -> bool:
        """Alias used by the JSONL manifest and by transformation gating."""
        return self.derivatives_allowed


@dataclass(frozen=True)
class Match:
    """One exact unique Echoes original to FMA track pairing."""

    original_audio: str
    track_id: int
    title: str
    artist: str
    subset: str
    split: str
    license_title: str
    duration: float | None
    genre_top: str

    @property
    def content_group(self) -> str:
        """Stable group shared by the original and every fake derived from it."""
        return content_group_for(self.track_id)

    def member_path(self, prefix: str = FMA_AUDIO_PREFIX) -> str:
        return fma_member_path(self.track_id, prefix=prefix)


@dataclass
class ExtractionReport:
    """Counts and diagnostics for one extraction run."""

    subset: str
    matches_selected: int = 0
    reals_written: int = 0
    reals_skipped_existing: int = 0
    fakes_written: int = 0
    fakes_skipped_existing: int = 0
    nd_originals: int = 0
    missing_members: list[str] = field(default_factory=list)
    checksum_mismatches: list[str] = field(default_factory=list)
    rows: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "subset": self.subset,
            "matches_selected": self.matches_selected,
            "reals_written": self.reals_written,
            "reals_skipped_existing": self.reals_skipped_existing,
            "fakes_written": self.fakes_written,
            "fakes_skipped_existing": self.fakes_skipped_existing,
            "nd_originals": self.nd_originals,
            "manifest_rows": len(self.rows),
            "missing_members": self.missing_members,
            "checksum_mismatches": self.checksum_mismatches,
        }


def content_group_for(track_id: int) -> str:
    """Group id tying one bona fide original to all fakes sharing its content."""
    return "fma-{:06d}".format(int(track_id))


def fma_member_path(track_id: int, prefix: str = FMA_AUDIO_PREFIX) -> str:
    """Derive the FMA archive member path for a track id.

    FMA stores audio as <prefix>/<first three digits>/<six digits>.mp3, so the
    path is computed from the id and never requires listing the archive.
    """
    track = int(track_id)
    if track < 0:
        raise ValueError("track_id must be non-negative, got {!r}".format(track_id))
    return "{}/{:03d}/{:06d}.mp3".format(prefix, track // 1000, track)


def safe_member_name(name: str) -> PurePosixPath:
    """Validate an archive member name, rejecting zip-slip style payloads."""
    normalized = str(name).replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        raise UnsafeMemberError("member is not a file: {!r}".format(name))
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeMemberError("member path is absolute: {!r}".format(name))
    parts = PurePosixPath(normalized).parts
    if any(part == ".." for part in parts):
        raise UnsafeMemberError("member path escapes its root: {!r}".format(name))
    return PurePosixPath(normalized)


def resolve_output_path(output_root: str | Path, relative_path: str) -> Path:
    """Resolve relative_path under output_root, refusing any escape."""
    root = Path(output_root).resolve()
    candidate = (root / safe_member_name(relative_path)).resolve()
    if candidate == root or root not in candidate.parents:
        raise UnsafeMemberError(
            "destination escapes the output root: {!r}".format(relative_path)
        )
    return candidate


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()


def classify_license(title: str, url: str) -> LicenseTerms:
    """Normalize an FMA licence title/URL pair into machine-checkable terms.

    The URL is authoritative because FMA titles are free text with many
    jurisdiction spellings. ND detection is deliberately conservative: any hint
    of "no derivatives" in either field disables transformations.
    """
    raw_title = (title or "").strip()
    raw_url = (url or "").strip()
    normalized_url = raw_url.replace("http://", "https://")
    lowered_url = normalized_url.lower()
    lowered_title = raw_title.lower()

    no_derivatives = bool(
        re.search(r"/by(?:-nc)?-nd(?:[/-]|$)", lowered_url)
        or re.search(r"no[\s-]*deriv", lowered_title)
    )
    non_commercial = bool(
        re.search(r"/by-nc", lowered_url)
        or "noncommercial" in lowered_title.replace("-", "").replace(" ", "")
    )

    license_id = ""
    if "publicdomain/zero" in lowered_url or "cc0" in lowered_title:
        license_id = "CC0-1.0"
    elif "publicdomain/mark" in lowered_url or "public domain mark" in lowered_title:
        license_id = "CC-PDM-1.0"
    elif "publicdomain" in lowered_url or lowered_title == "public domain":
        license_id = "CC-PD"
    else:
        match = re.search(r"/licenses/(by[a-z-]*)/(\d+\.\d+)(?:/([a-z]{2}))?", lowered_url)
        if match:
            clause, version, jurisdiction = match.groups()
            license_id = "CC-{}-{}".format(clause.upper(), version)
            if jurisdiction:
                license_id = "{}-{}".format(license_id, jurisdiction.upper())
        elif raw_title:
            license_id = _slug(raw_title).upper()

    if not license_id:
        raise MatchSelectionError(
            "cannot classify licence: title={!r} url={!r}".format(title, url)
        )
    if not normalized_url:
        raise MatchSelectionError(
            "licence {} is missing a licence URL".format(license_id)
        )

    # Every licence in this corpus (CC-BY family, ND variants, public domain)
    # permits verbatim redistribution with attribution. ND removes only the
    # right to distribute modified versions.
    return LicenseTerms(
        license_id=license_id,
        license_url=normalized_url,
        redistribution_allowed=True,
        derivatives_allowed=not no_derivatives,
        commercial_use_allowed=not non_commercial,
        raw_title=raw_title,
    )


def load_matches(matches_csv: str | Path, subset: str | None = "small") -> list[Match]:
    """Load exact unique matches from the data-audit CSV, filtered by subset.

    tools/analyze_echoes_fma_matches.py only writes rows it resolved to exactly
    one FMA track (ambiguous and unmatched originals go to the JSON report), so
    every CSV row here is already an exact unique match. Duplicate track ids are
    collapsed defensively and output is sorted for deterministic runs.
    """
    path = Path(matches_csv)
    if not path.is_file():
        raise MatchSelectionError(
            "match audit not found: {}. Run tools/analyze_echoes_fma_matches.py first.".format(path)
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"original_audio", "track_id", "subset", "license"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise MatchSelectionError(
                "match audit is missing columns: {}".format(sorted(missing))
            )
        rows = list(reader)

    selected: dict[int, Match] = {}
    for row in rows:
        row_subset = (row.get("subset") or "").strip()
        if subset is not None and row_subset != subset:
            continue
        raw_id = str(row.get("track_id") or "").strip()
        try:
            track_id = int(raw_id)
        except ValueError as exc:
            raise MatchSelectionError(
                "invalid track_id in match audit: {!r}".format(raw_id)
            ) from exc
        if track_id in selected:
            continue
        raw_duration = str(row.get("duration") or "").strip()
        try:
            duration = float(raw_duration) if raw_duration else None
        except ValueError:
            duration = None
        selected[track_id] = Match(
            original_audio=(row.get("original_audio") or "").strip(),
            track_id=track_id,
            title=(row.get("title") or "").strip(),
            artist=(row.get("artist") or "").strip(),
            subset=row_subset,
            split=(row.get("split") or "").strip(),
            license_title=(row.get("license") or "").strip(),
            duration=duration,
            genre_top=(row.get("genre_top") or "").strip(),
        )
    if not selected:
        raise MatchSelectionError(
            "no matches found for subset={!r} in {}".format(subset, path)
        )
    return [selected[key] for key in sorted(selected)]


def load_fma_license_index(
    raw_tracks_csv: str | Path, track_ids: set[int]
) -> dict[int, dict[str, str]]:
    """Read licence/attribution facts for track_ids from FMA raw_tracks.csv.

    Streams the 122 MB file with the csv module and keeps only the wanted ids, so
    peak memory stays proportional to the subset rather than the whole corpus.
    """
    path = Path(raw_tracks_csv)
    if not path.is_file():
        raise MatchSelectionError("FMA raw_tracks.csv not found: {}".format(path))
    wanted = {int(value) for value in track_ids}
    index: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for column in ("track_id", "license_title", "license_url", "track_url", "artist_name"):
            if column not in (reader.fieldnames or ()):
                raise MatchSelectionError(
                    "raw_tracks.csv is missing column {!r}".format(column)
                )
        for row in reader:
            raw_id = str(row.get("track_id") or "").strip()
            if not raw_id.isdigit():
                continue
            track_id = int(raw_id)
            if track_id not in wanted:
                continue
            index[track_id] = {
                "license_title": (row.get("license_title") or "").strip(),
                "license_url": (row.get("license_url") or "").strip(),
                "track_url": (row.get("track_url") or "").strip(),
                "artist_name": (row.get("artist_name") or "").strip(),
                "album_title": (row.get("album_title") or "").strip(),
                "artist_id": (row.get("artist_id") or "").strip(),
            }
            if len(index) == len(wanted):
                break
    return index


def load_echoes_fakes(
    manifest_csv_bytes: bytes, originals: set[str]
) -> dict[str, list[dict[str, str]]]:
    """Group Echoes manifest rows by original_audio for the given originals.

    Rows are keyed by original_audio and sorted by path_in_dataset so extraction
    order is deterministic.
    """
    text = manifest_csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(text.splitlines())
    for column in ("path_in_dataset", "original_audio", "generator", "type"):
        if column not in (reader.fieldnames or ()):
            raise MatchSelectionError(
                "Echoes manifest is missing column {!r}".format(column)
            )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in reader:
        original = (row.get("original_audio") or "").strip()
        if original not in originals:
            continue
        grouped.setdefault(original, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: str(item.get("path_in_dataset") or ""))
    return grouped


def probe_audio(path: str | Path) -> dict[str, object]:
    """Read codec facts from the first MPEG frame without third-party deps.

    Returns codec, sample_rate, channels, bitrate_kbps and a best-effort
    duration_seconds (exact for Xing/Info VBR headers, CBR estimate otherwise).
    Unparseable input yields empty values instead of raising, so one odd file
    cannot abort a whole run.
    """
    data = Path(path).read_bytes()
    unknown: dict[str, object] = {
        "codec": "",
        "sample_rate": "",
        "channels": "",
        "bitrate_kbps": "",
        "duration_seconds": "",
    }
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return _probe_wav(data)
    offset = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        offset = 10 + size
    limit = min(len(data) - 4, offset + 1_000_000)
    while 0 <= offset < limit:
        if data[offset] != 0xFF or (data[offset + 1] & 0xE0) != 0xE0:
            offset += 1
            continue
        header = data[offset : offset + 4]
        version_bits = (header[1] >> 3) & 0x03
        layer_bits = (header[1] >> 1) & 0x03
        bitrate_index = (header[2] >> 4) & 0x0F
        rate_index = (header[2] >> 2) & 0x03
        mode = (header[3] >> 6) & 0x03
        if version_bits == 1 or layer_bits != 1 or rate_index == 3 or bitrate_index in {0, 15}:
            offset += 1
            continue
        sample_rate = _SAMPLE_RATES[version_bits][rate_index]
        table = _BITRATES_V1_L3 if version_bits == 3 else _BITRATES_V2_L3
        bitrate = table[bitrate_index]
        channels = 1 if mode == 3 else 2
        samples_per_frame = 1152 if version_bits == 3 else 576
        duration: object = ""
        window = data[offset : offset + 256]
        marker = max(window.find(b"Xing"), window.find(b"Info"))
        if marker != -1:
            cursor = offset + marker + 4
            if len(data) >= cursor + 8:
                flags = int.from_bytes(data[cursor : cursor + 4], "big")
                if flags & 0x1:
                    frames = int.from_bytes(data[cursor + 4 : cursor + 8], "big")
                    if frames:
                        duration = round(frames * samples_per_frame / sample_rate, 3)
        if duration == "" and bitrate:
            duration = round((len(data) - offset) * 8 / (bitrate * 1000), 3)
        return {
            "codec": "mp3",
            "sample_rate": sample_rate,
            "channels": channels,
            "bitrate_kbps": bitrate,
            "duration_seconds": duration,
        }
    return unknown


def _probe_wav(data: bytes) -> dict[str, object]:
    """Parse the fmt/data chunks of a RIFF WAVE header."""
    result: dict[str, object] = {
        "codec": "wav",
        "sample_rate": "",
        "channels": "",
        "bitrate_kbps": "",
        "duration_seconds": "",
    }
    cursor = 12
    channels = sample_rate = bits = 0
    while cursor + 8 <= len(data):
        chunk_id = data[cursor : cursor + 4]
        size = int.from_bytes(data[cursor + 4 : cursor + 8], "little")
        body = data[cursor + 8 : cursor + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            channels = int.from_bytes(body[2:4], "little")
            sample_rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            result["channels"] = channels
            result["sample_rate"] = sample_rate
        elif chunk_id == b"data" and channels and sample_rate and bits:
            frame = channels * bits // 8
            if frame:
                result["duration_seconds"] = round(len(body) / frame / sample_rate, 3)
            result["bitrate_kbps"] = sample_rate * channels * bits // 1000
            break
        cursor += 8 + size + (size % 2)
    return result


def read_archive_checksums(
    archive: zipfile.ZipFile, prefix: str = FMA_AUDIO_PREFIX
) -> dict[str, str]:
    """Read the FMA checksums member, mapping member path to SHA-1 digest."""
    try:
        payload = archive.read("{}/checksums".format(prefix))
    except KeyError:
        return {}
    checksums: dict[str, str] = {}
    for line in payload.decode("utf-8", errors="replace").splitlines():
        digest, _, relative = line.partition("  ")
        relative = relative.strip()
        if digest.strip() and relative:
            checksums["{}/{}".format(prefix, relative)] = digest.strip().lower()
    return checksums


def open_archive(archive_path: str | Path, label: str) -> zipfile.ZipFile:
    """Open an archive, diagnosing partial downloads instead of crashing."""
    path = Path(archive_path)
    if not path.is_file():
        raise ArchiveIncompleteError("{} archive not found: {}".format(label, path))
    size = path.stat().st_size
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ArchiveIncompleteError(
            "{} archive at {} is not a readable zip ({}). Current size {:,} bytes; "
            "the download is probably still in flight. Wait for it to complete, or "
            "pass an already extracted directory instead.".format(label, path, exc, size)
        ) from exc


def _blank_row() -> dict[str, str]:
    return {column: "" for column in REQUIRED_COLUMNS}


def build_real_row(
    match: Match,
    relative_path: str,
    sha256: str,
    audio: dict[str, object],
    terms: LicenseTerms,
    fma_meta: dict[str, str],
    source_version: str,
) -> dict[str, str]:
    """Canonical manifest row for a bona fide FMA original."""
    artist = fma_meta.get("artist_name") or match.artist
    track_url = fma_meta.get("track_url") or FMA_SOURCE_URL
    attribution = "{} - {} ({}), licensed {}".format(
        artist or "unknown artist", match.title or "untitled", track_url, terms.license_id
    )
    row = _blank_row()
    row.update(
        sample_id="fma-{:06d}-original".format(match.track_id),
        relative_path=relative_path,
        sha256=sha256,
        source_dataset="fma_{}".format(match.subset or "small"),
        source_version=source_version,
        source_url=track_url,
        license_id=terms.license_id,
        license_url=terms.license_url,
        redistribution_allowed="true" if terms.redistribution_allowed else "false",
        derivatives_allowed="true" if terms.derivatives_allowed else "false",
        attribution=attribution,
        source_family="fma",
        content_id=match.content_group,
        codec=str(audio.get("codec") or ""),
        sample_rate=str(audio.get("sample_rate") or ""),
        channels=str(audio.get("channels") or ""),
        duration_seconds=str(audio.get("duration_seconds") or ""),
        transformations_json="[]",
        **REAL_LABELS,
    )
    return row


def build_fake_row(
    match: Match,
    fake: dict[str, str],
    relative_path: str,
    sha256: str,
    audio: dict[str, object],
    parent_sha256: str,
    echoes_version: str,
) -> dict[str, str]:
    """Canonical manifest row for one Echoes fake paired with the original.

    parent_sha256 records the paired bona fide reference, and content_id ties the
    fake to its group. That is a semantic pairing, not a claim that the audio was
    derived by editing the FMA recording: TTA fakes are generated from a text
    description alone, and only ATA fakes are audio-conditioned. transformations
    stays empty because no transformation is applied by this tool.
    """
    generator = (fake.get("generator") or "").strip()
    generation_type = (fake.get("type") or "").strip().upper()
    path_in_dataset = (fake.get("path_in_dataset") or "").strip()
    stem = PurePosixPath(path_in_dataset).stem or _slug(path_in_dataset)
    conditioning = "audio-and-text" if generation_type == "ATA" else "text-only"
    row = _blank_row()
    row.update(
        sample_id="echoes-{}".format(_slug(stem)),
        relative_path=relative_path,
        sha256=sha256,
        source_dataset="echoes",
        source_version=echoes_version,
        source_url=ECHOES_SOURCE_URL,
        license_id=ECHOES_LICENSE_ID,
        license_url=ECHOES_LICENSE_URL,
        redistribution_allowed="true",
        derivatives_allowed="true",
        attribution=(
            "Echoes dataset (CC BY-SA 4.0); {} {} generation referencing {}".format(
                generator or "unknown generator", generation_type or "unknown-type",
                match.original_audio or match.title,
            )
        ),
        generator_family=generator,
        generator_model=generator,
        source_family="echoes",
        content_id=match.content_group,
        codec=str(audio.get("codec") or ""),
        sample_rate=str(audio.get("sample_rate") or ""),
        channels=str(audio.get("channels") or ""),
        duration_seconds=str(audio.get("duration_seconds") or ""),
        parent_sha256=parent_sha256 if conditioning == "audio-and-text" else "",
        transformations_json="[]",
        **FAKE_LABELS,
    )
    return row


def _sidecar(
    row: dict[str, str],
    content_group: str,
    transformation_allowed: bool,
    extra: dict[str, object],
) -> dict[str, object]:
    """Merge a canonical row with provenance fields the CSV schema cannot hold."""
    record: dict[str, object] = dict(row)
    record["content_group"] = content_group
    record["transformation_allowed"] = transformation_allowed
    record.update(extra)
    return record


def write_manifests(
    rows: list[dict[str, object]], csv_path: str | Path, jsonl_path: str | Path
) -> None:
    """Write the canonical CSV plus the richer JSONL sidecar deterministically."""
    csv_file = Path(csv_path)
    jsonl_file = Path(jsonl_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REQUIRED_COLUMNS})
    with jsonl_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _extract_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    overwrite: bool,
) -> tuple[bool, int]:
    """Copy one archive member verbatim to destination.

    Returns (written, size). An existing file of identical size is left alone so
    reruns are cheap and idempotent.
    """
    info = archive.getinfo(member)
    if destination.exists() and not overwrite and destination.stat().st_size == info.file_size:
        return False, info.file_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with archive.open(info) as source, temporary.open("wb") as sink:
        while chunk := source.read(1024 * 1024):
            sink.write(chunk)
    temporary.replace(destination)
    return True, info.file_size


def extract_subset(
    matches_csv: str | Path,
    fma_archive: str | Path,
    echoes_archive: str | Path,
    raw_tracks_csv: str | Path,
    output_root: str | Path,
    subset: str = "small",
    fma_version: str = "fma_small",
    echoes_version: str = "unknown",
    limit: int | None = None,
    overwrite: bool = False,
    verify_checksums: bool = True,
) -> ExtractionReport:
    """Extract only the matched originals plus their fakes, then emit manifests.

    Work is planned from the match audit and the FMA track ids, so the number of
    members touched is bounded by the match count regardless of archive size.
    """
    matches = load_matches(matches_csv, subset=subset)
    if limit is not None:
        matches = matches[:limit]
    report = ExtractionReport(subset=subset, matches_selected=len(matches))

    license_index = load_fma_license_index(raw_tracks_csv, {m.track_id for m in matches})
    root = Path(output_root).resolve()

    # Open both archives before creating anything, so a partial download or an
    # unreadable archive leaves the output root untouched.
    with open_archive(fma_archive, "FMA") as fma, open_archive(echoes_archive, "Echoes") as echoes:
        root.mkdir(parents=True, exist_ok=True)
        checksums = read_archive_checksums(fma) if verify_checksums else {}
        manifest_member = "{}/dataset_manifest.csv".format(ECHOES_MEMBER_PREFIX)
        try:
            manifest_bytes = echoes.read(manifest_member)
        except KeyError as exc:
            raise ArchiveIncompleteError(
                "Echoes archive is missing {}".format(manifest_member)
            ) from exc
        fakes_by_original = load_echoes_fakes(
            manifest_bytes, {m.original_audio for m in matches}
        )
        fma_names = set(fma.namelist())
        echoes_names = set(echoes.namelist())

        for match in matches:
            member = match.member_path()
            safe_member_name(member)
            if member not in fma_names:
                report.missing_members.append(member)
                continue
            meta = license_index.get(match.track_id, {})
            terms = classify_license(
                meta.get("license_title") or match.license_title,
                meta.get("license_url", ""),
            )
            if not terms.derivatives_allowed:
                report.nd_originals += 1

            relative = "real/fma/{:03d}/{:06d}.mp3".format(match.track_id // 1000, match.track_id)
            destination = resolve_output_path(root, relative)
            written, _ = _extract_member(fma, member, destination, overwrite)
            if written:
                report.reals_written += 1
            else:
                report.reals_skipped_existing += 1

            expected_sha1 = checksums.get(member)
            if expected_sha1:
                digest = hashlib.sha1()
                with destination.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha1:
                    report.checksum_mismatches.append(member)

            real_sha256 = sha256_file(destination)
            real_audio = probe_audio(destination)
            real_row = build_real_row(
                match, relative, real_sha256, real_audio, terms, meta, fma_version
            )
            report.rows.append(
                _sidecar(
                    real_row,
                    match.content_group,
                    terms.derivatives_allowed,
                    {
                        "role": "bona_fide_original",
                        "fma_track_id": match.track_id,
                        "fma_subset": match.subset,
                        "fma_split": match.split,
                        "fma_genre_top": match.genre_top,
                        "license_title_raw": terms.raw_title,
                        "commercial_use_allowed": terms.commercial_use_allowed,
                        "archive_member": member,
                        "archive_sha1": expected_sha1 or "",
                        "echoes_original_audio": match.original_audio,
                    },
                )
            )

            for fake in fakes_by_original.get(match.original_audio, []):
                path_in_dataset = (fake.get("path_in_dataset") or "").strip()
                if not path_in_dataset:
                    continue
                fake_member = "{}/{}".format(ECHOES_MEMBER_PREFIX, path_in_dataset)
                safe_member_name(fake_member)
                if fake_member not in echoes_names:
                    report.missing_members.append(fake_member)
                    continue
                fake_relative = "fake/echoes/{}".format(safe_member_name(path_in_dataset).as_posix())
                fake_destination = resolve_output_path(root, fake_relative)
                fake_written, _ = _extract_member(
                    echoes, fake_member, fake_destination, overwrite
                )
                if fake_written:
                    report.fakes_written += 1
                else:
                    report.fakes_skipped_existing += 1
                fake_audio = probe_audio(fake_destination)
                fake_row = build_fake_row(
                    match,
                    fake,
                    fake_relative,
                    sha256_file(fake_destination),
                    fake_audio,
                    real_sha256,
                    echoes_version,
                )
                generation_type = (fake.get("type") or "").strip().upper()
                report.rows.append(
                    _sidecar(
                        fake_row,
                        match.content_group,
                        True,
                        {
                            "role": "ai_generated_pair",
                            "generation_type": generation_type,
                            "conditioning": (
                                "audio-and-text" if generation_type == "ATA" else "text-only"
                            ),
                            "prompt_description": (fake.get("description") or "").strip(),
                            "echoes_genre": (fake.get("genre") or "").strip(),
                            "echoes_path_in_dataset": path_in_dataset,
                            "echoes_original_audio": match.original_audio,
                            "paired_real_sha256": real_sha256,
                            "paired_real_transformation_allowed": terms.derivatives_allowed,
                            "reported_duration_seconds": (fake.get("duration") or "").strip(),
                        },
                    )
                )
    return report
