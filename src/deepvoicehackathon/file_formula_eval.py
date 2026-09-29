"""Offline FILE_FAKE formula research on cached DF-Arena intermediate scores.

Why this module exists
----------------------
The official baseline recomputes every model forward pass for each FILE_FAKE
variant, so comparing N candidate formulas costs N full GPU passes. Yet every
candidate the submission program cares about is a pure function of the same
small set of intermediates:

* PANNs voice/music presence probabilities (one pair per file),
* per-segment DF-Arena spoof probabilities for the HTDemucs voice stem,
* the same for the HTDemucs music stem,
* the same for the raw 16 kHz mixture.

This module extracts those once, caches them keyed by audio SHA-256 plus an
exact pipeline-config fingerprint, then evaluates an arbitrary formula grid
from cache with no model loaded.

Fidelity to baseline/official/script.py
---------------------------------------
* Segmentation reuses baseline_segment_starts / baseline_extract_segment from
  deepvoicehackathon.music_eval, which are verbatim transcriptions of
  get_segment_starts / extract_segment.
* predict_fake short-circuits to 0.0 when stream RMS < SILENCE_RMS. A silent
  stream is therefore cached with an empty segment list and silent=True, and
  every pooling of a silent stream returns 0.0. The silence branch is stored,
  not re-derived per formula.
* separate_voice_and_music normalises by mono mean/std, sums non-vocal stems,
  resamples to 16 kHz, and returns matched zero vectors when std < 1e-8.
* PANNs presence is the per-label-group max over 32 kHz segments.

Research-only scope
-------------------
Nothing here runs at submission time. Ranking uses set-level statistics
(bootstrap intervals, leave-one-family-out worst case), but a per-file score
never depends on another file: evaluate_formula consumes exactly one record.
See DOMAIN_LIMITATION for why the local ranking is weak evidence about the
official mixed-domain test set.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .music_eval import (
    BASELINE_SEGMENT_SAMPLES,
    BASELINE_SILENCE_RMS,
    ManifestColumns,
    ManifestRow,
    baseline_extract_segment,
    baseline_segment_starts,
    config_fingerprint,
    crop_or_pad,
    file_digest,
    grouped_bootstrap_eer_ci,
    leave_one_generator_family_out,
    load_manifest,
    sample_rows,
    summarize,
)

# Baseline constants (baseline/official/script.py).
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = BASELINE_SEGMENT_SAMPLES
SILENCE_RMS = BASELINE_SILENCE_RMS
NEUTRAL_PRESENCE = 0.5

STREAMS = ("voice", "music", "mixture")
POOLING_KINDS = ("max", "mean", "topk", "mix")
COMBINE_MODES = ("max", "prob_or", "sum_clipped", "mean")
SELECTION_METRICS = ("worst_family", "blend", "overall")

DOMAIN_LIMITATION = (
    "Paired-music domain only. This corpus is duration-controlled music with "
    "MUSIC_PRESENT=1 on every row and no voice-only or speech-only items, so the "
    "local FILE_FAKE label is effectively the music label. A formula that wins "
    "here is evidence about the music half of the official distribution and about "
    "pooling mechanics; it is not evidence about voice-dominant or mixed files. "
    "Presence gating is close to inert locally because PANNs music presence "
    "saturates near 1.0, so gating exponents must be judged on the official set."
)

DEFAULT_MANIFEST_COLUMNS = ManifestColumns(
    rel_path="relative_path",
    audio_path="audio_path",
    label="file_fake",
    group="content_id",
    generator_family="generator_family",
    license_id="license_id",
    transform="transformations_json",
)


class ExtractionUnavailableError(RuntimeError):
    """Raised when extraction dependencies or checkpoints are missing."""


class ExtractionFailure(RuntimeError):
    """Raised when one file cannot be processed."""


# ---------------------------------------------------------------------------
# 1. Pipeline configuration and cache identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionConfig:
    """Exact pipeline configuration a cached record belongs to.

    Every field that can change a cached number joins the fingerprint,
    including device (CPU and CUDA kernels do not agree bit for bit) and
    pipeline_version (bump when extraction semantics change). weight_digests is
    empty by default because hashing multi-GB checkpoints every run is wasteful;
    pass digests explicitly when a checkpoint swap must invalidate the cache.
    """

    device: str = "cpu"
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    panns_sample_rate: int = PANNS_SAMPLE_RATE
    segment_samples: int = SEGMENT_SAMPLES
    silence_rms: float = SILENCE_RMS
    df_checkpoint: str = "df_arena_1b"
    htdemucs_checkpoint: str = "955717e8-8726e21a.th"
    panns_checkpoint: str = "Cnn14_mAP=0.431.pth"
    demucs_shifts: int = 0
    demucs_split: bool = True
    demucs_overlap: float = 0.25
    max_audio_seconds: float | None = None
    crop_mode: str = "dataset_eval"
    resample_type: str = "soxr_hq"
    pipeline_version: int = 1
    weight_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.segment_samples <= 0:
            raise ValueError("segment_samples must be positive")
        if self.audio_sample_rate <= 0 or self.panns_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if self.silence_rms < 0.0:
            raise ValueError("silence_rms must be non-negative")
        if self.max_audio_seconds is not None and self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")

    @property
    def max_audio_samples(self) -> int | None:
        if self.max_audio_seconds is None:
            return None
        return int(round(float(self.max_audio_seconds) * self.audio_sample_rate))

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["weight_digests"] = {name: digest for name, digest in self.weight_digests}
        return payload

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex digest of the whole configuration."""
        return config_fingerprint(self.as_dict())

    def cache_key(self, audio_sha256: str) -> str:
        """Cache key for one audio file under this configuration."""
        token = str(audio_sha256).strip().lower()
        if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
            raise ValueError(f"audio_sha256 must be a hex SHA-256 digest, got {audio_sha256!r}")
        return f"{self.fingerprint}:{token}"


# ---------------------------------------------------------------------------
# 2. Cached intermediate record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamScores:
    """Per-segment DF-Arena spoof probabilities for one audio stream.

    silent records the baseline RMS short-circuit: the baseline returns 0.0
    without running the model, so segments is empty and every pooling yields
    0.0.
    """

    segments: tuple[float, ...] = ()
    silent: bool = False
    rms: float = 0.0
    n_samples: int = 0

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.segments)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("segment scores must be finite")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("segment scores must lie in [0, 1]")
        object.__setattr__(self, "segments", values)
        if self.silent and values:
            raise ValueError("a silent stream must not carry segment scores")
        if not self.silent and not values:
            raise ValueError("a non-silent stream must carry at least one segment score")

    def as_dict(self) -> dict:
        return {
            "segments": [float(value) for value in self.segments],
            "silent": bool(self.silent),
            "rms": float(self.rms),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "StreamScores":
        return cls(
            segments=tuple(float(value) for value in payload.get("segments", ())),
            silent=bool(payload.get("silent", False)),
            rms=float(payload.get("rms", 0.0)),
            n_samples=int(payload.get("n_samples", 0)),
        )

    @classmethod
    def silence(cls, rms: float = 0.0, n_samples: int = 0) -> "StreamScores":
        return cls(segments=(), silent=True, rms=float(rms), n_samples=int(n_samples))


@dataclass(frozen=True)
class IntermediateRecord:
    """Everything a FILE_FAKE formula needs for one file."""

    sample_id: str
    audio_sha256: str
    config_fingerprint: str
    label: int
    group: str
    generator_family: str
    license_id: str
    transform: str
    voice_present: float
    music_present: float
    voice: StreamScores
    music: StreamScores
    mixture: StreamScores
    duration_seconds: float = 0.0
    presence_failed: bool = False
    seconds: float = 0.0

    def stream(self, name: str) -> StreamScores:
        if name not in STREAMS:
            raise ValueError(f"stream must be one of {STREAMS}")
        return getattr(self, name)

    def as_dict(self) -> dict:
        payload = {
            "sample_id": self.sample_id,
            "audio_sha256": self.audio_sha256,
            "config_fingerprint": self.config_fingerprint,
            "label": int(self.label),
            "group": self.group,
            "generator_family": self.generator_family,
            "license_id": self.license_id,
            "transform": self.transform,
            "voice_present": float(self.voice_present),
            "music_present": float(self.music_present),
            "duration_seconds": float(self.duration_seconds),
            "presence_failed": bool(self.presence_failed),
            "seconds": float(self.seconds),
        }
        for name in STREAMS:
            payload[name] = self.stream(name).as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "IntermediateRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            audio_sha256=str(payload["audio_sha256"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            label=int(payload["label"]),
            group=str(payload.get("group", "unknown")),
            generator_family=str(payload.get("generator_family", "unknown")),
            license_id=str(payload.get("license_id", "unknown")),
            transform=str(payload.get("transform", "unknown")),
            voice_present=float(payload["voice_present"]),
            music_present=float(payload["music_present"]),
            voice=StreamScores.from_dict(payload["voice"]),
            music=StreamScores.from_dict(payload["music"]),
            mixture=StreamScores.from_dict(payload["mixture"]),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            presence_failed=bool(payload.get("presence_failed", False)),
            seconds=float(payload.get("seconds", 0.0)),
        )

    @property
    def cache_key(self) -> str:
        return f"{self.config_fingerprint}:{self.audio_sha256}"


REQUIRED_RECORD_KEYS = (
    "sample_id",
    "audio_sha256",
    "config_fingerprint",
    "label",
    "voice_present",
    "music_present",
    "voice",
    "music",
    "mixture",
)


# ---------------------------------------------------------------------------
# 3. Append-only JSONL cache with safe resume
# ---------------------------------------------------------------------------

def write_json_atomic(path, payload) -> Path:
    """Write JSON through a same-directory temp file plus os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


class IntermediateStore:
    """Resumable JSONL store of IntermediateRecord values.

    Records append one JSON object per line and flush with fsync, so an
    interrupted run loses at most the record being written. Loading skips
    unparsable or incomplete lines and counts them in corrupt_lines instead of
    failing, which makes resume safe after a hard kill. Later lines win for a
    repeated key, so re-extraction overrides a stale value without a rewrite;
    compact() rewrites the file atomically when that history is no longer
    wanted.
    """

    def __init__(self, path=None, enabled: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = bool(enabled and self.path is not None)
        self._records: dict[str, IntermediateRecord] = {}
        self.corrupt_lines = 0
        self.hits = 0
        self.misses = 0
        self.appended = 0
        if self.path is not None and self.path.is_file():
            self.load()

    def load(self) -> None:
        """Read every valid line, tolerating a truncated tail."""
        self._records.clear()
        self.corrupt_lines = 0
        if self.path is None or not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    self.corrupt_lines += 1
                    continue
                if not isinstance(payload, dict) or any(key not in payload for key in REQUIRED_RECORD_KEYS):
                    self.corrupt_lines += 1
                    continue
                try:
                    record = IntermediateRecord.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    self.corrupt_lines += 1
                    continue
                self._records[record.cache_key] = record

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._records

    def get(self, key: str) -> IntermediateRecord | None:
        record = self._records.get(str(key))
        if record is None:
            self.misses += 1
            return None
        self.hits += 1
        return record

    def records(self) -> list[IntermediateRecord]:
        """All records ordered by sample id then key, for deterministic output."""
        return sorted(self._records.values(), key=lambda item: (item.sample_id, item.cache_key))

    def append(self, record: IntermediateRecord) -> None:
        """Append one record durably and update the in-memory view."""
        self._records[record.cache_key] = record
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.appended += 1

    def compact(self) -> None:
        """Atomically rewrite the file with one deduplicated line per key."""
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle_fd, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
                for record in self.records():
                    handle.write(json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        self.corrupt_lines = 0


# ---------------------------------------------------------------------------
# 4. Formula math: segment pooling, component combination, presence gating
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolingSpec:
    """How to reduce one stream's segment scores to a single component score.

    kind semantics:

    * max  - baseline sub01/sub05 music pooling.
    * mean - baseline sub02/sub04/sub06 pooling.
    * topk - mean of the k highest segment scores; k=1 equals max and
      k>=n_segments equals mean, so topk spans both anchors continuously.
    * mix  - convex blend alpha*mean + (1-alpha)*max, i.e. alpha=0 is max and
      alpha=1 is mean.

    Both interpolating kinds are monotone in their parameter and reduce exactly
    to the anchors at the endpoints, which keeps every swept variant comparable
    to a submitted formula.
    """

    kind: str = "max"
    k: int = 1
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in POOLING_KINDS:
            raise ValueError(f"pooling kind must be one of {POOLING_KINDS}")
        if self.kind == "topk" and self.k < 1:
            raise ValueError("topk pooling requires k >= 1")
        if self.kind == "mix" and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("mix pooling requires alpha in [0, 1]")

    @property
    def name(self) -> str:
        if self.kind == "topk":
            return f"top{self.k}"
        if self.kind == "mix":
            return f"mix{self.alpha:g}"
        return self.kind

    def as_dict(self) -> dict:
        return {"kind": self.kind, "k": int(self.k), "alpha": float(self.alpha), "name": self.name}


def pool_segments(segments: Sequence[float] | np.ndarray, spec: PoolingSpec | None = None) -> float:
    """Reduce per-segment spoof probabilities to one component score.

    An empty sequence returns 0.0, matching the baseline silence short-circuit
    in predict_fake, so a silent stream never contributes a positive score.
    """
    pooling = spec or PoolingSpec()
    values = np.asarray(list(segments), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if not np.isfinite(values).all():
        raise ValueError("segment scores must be finite")
    if pooling.kind == "max":
        return float(values.max())
    if pooling.kind == "mean":
        return float(values.mean())
    if pooling.kind == "topk":
        k = min(int(pooling.k), int(values.size))
        # Sort descending for determinism; partition order is unspecified.
        ordered = np.sort(values)[::-1]
        return float(ordered[:k].mean())
    return float(pooling.alpha * values.mean() + (1.0 - pooling.alpha) * values.max())


def gate(component: float, presence: float, exponent: float) -> float:
    """Apply presence gating with an exponent.

    exponent=1 reproduces the baseline product presence*component; exponent=0
    disables gating; values in between soften it. Presence is clipped into
    [0, 1] first because PANNs output and the 0.5 failure fallback are both
    probabilities and a negative base would make a fractional power complex.
    """
    if exponent < 0.0:
        raise ValueError("presence exponent must be non-negative")
    if exponent == 0.0:
        return float(component)
    clipped = min(1.0, max(0.0, float(presence)))
    if exponent == 1.0:
        return float(clipped * component)
    return float((clipped ** exponent) * component)


def combine_components(voice_score: float, music_score: float, mode: str = "max") -> float:
    """Combine gated voice/music component scores into one file score.

    * max          - baseline sub01/sub04/sub06 behaviour.
    * prob_or       - 1-(1-v)(1-w), the noisy-or of two independent detections.
    * sum_clipped   - min(1, v+w), a harsher additive variant.
    * mean          - 0.5*(v+w), which unlike the others can be dragged down by
      a confidently-real component.

    All modes stay within [0, 1] for inputs in [0, 1], which the metric
    validation in deepvoicehackathon.metrics requires.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"combine mode must be one of {COMBINE_MODES}")
    first = float(voice_score)
    second = float(music_score)
    if mode == "max":
        return max(first, second)
    if mode == "prob_or":
        return float(1.0 - (1.0 - first) * (1.0 - second))
    if mode == "sum_clipped":
        return float(min(1.0, first + second))
    return float(0.5 * (first + second))


@dataclass(frozen=True)
class FormulaSpec:
    """One complete FILE_FAKE formula.

    Evaluation order, matching the baseline and then extending it:

    1. pool the voice, music and mixture segment arrays independently;
    2. gate the voice and music components by their presence probabilities
       raised to voice_exponent / music_exponent;
    3. combine the two gated components with combine;
    4. convex-blend the result with the pooled raw mixture score using
       mixture_weight (0 = components only, 1 = mixture only).

    The mixture term is deliberately blended last and ungated: the raw mixture
    never went through separation, so presence gating would double-count the
    same evidence.
    """

    name: str
    voice_pooling: PoolingSpec = PoolingSpec("mean")
    music_pooling: PoolingSpec = PoolingSpec("mean")
    mixture_pooling: PoolingSpec = PoolingSpec("max")
    combine: str = "max"
    voice_exponent: float = 1.0
    music_exponent: float = 1.0
    mixture_weight: float = 0.0
    anchor: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("formula name must not be empty")
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"combine mode must be one of {COMBINE_MODES}")
        if not 0.0 <= self.mixture_weight <= 1.0:
            raise ValueError("mixture_weight must be in [0, 1]")
        if self.voice_exponent < 0.0 or self.music_exponent < 0.0:
            raise ValueError("presence exponents must be non-negative")

    @property
    def uses_mixture(self) -> bool:
        return self.mixture_weight > 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "anchor": self.anchor,
            "voice_pooling": self.voice_pooling.as_dict(),
            "music_pooling": self.music_pooling.as_dict(),
            "mixture_pooling": self.mixture_pooling.as_dict(),
            "combine": self.combine,
            "voice_exponent": float(self.voice_exponent),
            "music_exponent": float(self.music_exponent),
            "mixture_weight": float(self.mixture_weight),
        }

    @property
    def key(self) -> str:
        """Stable identity independent of the human-facing name."""
        payload = self.as_dict()
        payload.pop("name")
        payload.pop("anchor")
        return config_fingerprint(payload)


def evaluate_formula(record: IntermediateRecord, formula: FormulaSpec) -> float:
    """Score one file. Depends on that file's record only, never on the set."""
    voice_component = pool_segments(record.voice.segments, formula.voice_pooling)
    music_component = pool_segments(record.music.segments, formula.music_pooling)
    gated_voice = gate(voice_component, record.voice_present, formula.voice_exponent)
    gated_music = gate(music_component, record.music_present, formula.music_exponent)
    combined = combine_components(gated_voice, gated_music, formula.combine)
    if not formula.uses_mixture:
        return float(min(1.0, max(0.0, combined)))
    mixture_component = pool_segments(record.mixture.segments, formula.mixture_pooling)
    weight = float(formula.mixture_weight)
    blended = (1.0 - weight) * combined + weight * mixture_component
    return float(min(1.0, max(0.0, blended)))


# ---------------------------------------------------------------------------
# 5. Submitted formulas preserved as named anchors
# ---------------------------------------------------------------------------

ANCHOR_FORMULAS = (
    FormulaSpec(
        name="sub01_file_max_components",
        voice_pooling=PoolingSpec("max"),
        music_pooling=PoolingSpec("max"),
        combine="max",
        anchor="sub01",
    ),
    FormulaSpec(
        name="sub03_file_raw_mixture_max",
        mixture_pooling=PoolingSpec("max"),
        mixture_weight=1.0,
        anchor="sub03",
    ),
    FormulaSpec(
        name="sub04_file_mean_components",
        voice_pooling=PoolingSpec("mean"),
        music_pooling=PoolingSpec("mean"),
        combine="max",
        anchor="sub04",
    ),
    FormulaSpec(
        name="sub06_file_mean_components",
        voice_pooling=PoolingSpec("mean"),
        music_pooling=PoolingSpec("mean"),
        combine="max",
        anchor="sub06",
    ),
)
# sub06 is intentionally identical to sub04 here: sub06 changed only the
# submitted VOICE_FAKE output and explicitly kept sub04's FILE_FAKE formula.
# Both are listed so the ranking shows where the current official best sits.


def baseline_file_score(record: IntermediateRecord, pooling: str = "mean") -> float:
    """Verbatim baseline FILE_FAKE for cross-checking anchors.

    Mirrors combine_file_fake_score(predict_fake(voice), predict_fake(music))
    from baseline/official/script.py with a single pooling mode for both
    components, which is exactly what sub01 (max) and sub04/sub06 (mean) do.
    """
    if pooling not in ("max", "mean"):
        raise ValueError("pooling must be 'max' or 'mean'")
    voice = pool_segments(record.voice.segments, PoolingSpec(pooling))
    music = pool_segments(record.music.segments, PoolingSpec(pooling))
    return max(record.voice_present * voice, record.music_present * music)


# ---------------------------------------------------------------------------
# 6. Deterministic grid construction
# ---------------------------------------------------------------------------

DEFAULT_POOLINGS = (
    PoolingSpec("max"),
    PoolingSpec("mean"),
    PoolingSpec("topk", k=2),
    PoolingSpec("topk", k=3),
    PoolingSpec("mix", alpha=0.25),
    PoolingSpec("mix", alpha=0.5),
    PoolingSpec("mix", alpha=0.75),
)
DEFAULT_COMBINES = ("max", "prob_or", "sum_clipped")
DEFAULT_EXPONENTS = (0.0, 1.0)
DEFAULT_MIXTURE_WEIGHTS = (0.0, 0.25, 0.5)


def build_grid(
    poolings: Sequence[PoolingSpec] | None = None,
    combines: Sequence[str] | None = None,
    exponents: Sequence[float] | None = None,
    mixture_weights: Sequence[float] | None = None,
    mixture_poolings: Sequence[PoolingSpec] | None = None,
    tie_voice_music_pooling: bool = True,
    include_anchors: bool = True,
) -> list[FormulaSpec]:
    """Build the formula grid in a fixed, reproducible order.

    tie_voice_music_pooling keeps one pooling mode for both components, which
    is the identifiable comparison the submission program uses (sub04/sub06
    changed both together). Set it False to sweep the full cross product.

    Anchors come first and duplicate configurations are dropped by
    FormulaSpec.key, so an anchor is never re-listed under a generated name and
    the grid contains each distinct formula exactly once.
    """
    # None means "use the default axis"; an explicitly empty axis is an error,
    # because silently substituting defaults would hide a bad CLI invocation.
    pool_options = tuple(DEFAULT_POOLINGS if poolings is None else poolings)
    combine_options = tuple(DEFAULT_COMBINES if combines is None else combines)
    exponent_options = tuple(
        float(value) for value in (DEFAULT_EXPONENTS if exponents is None else exponents)
    )
    weight_options = tuple(
        float(value) for value in (DEFAULT_MIXTURE_WEIGHTS if mixture_weights is None else mixture_weights)
    )
    mixture_options = tuple(
        (PoolingSpec("max"), PoolingSpec("mean")) if mixture_poolings is None else mixture_poolings
    )
    if not pool_options or not combine_options or not exponent_options or not weight_options:
        raise ValueError("grid axes must not be empty")
    if not mixture_options:
        raise ValueError("grid axes must not be empty")

    grid: list[FormulaSpec] = []
    seen: set[str] = set()

    def add(formula: FormulaSpec, force: bool = False) -> None:
        if formula.key in seen and not force:
            return
        seen.add(formula.key)
        grid.append(formula)

    if include_anchors:
        for anchor in ANCHOR_FORMULAS:
            # force: sub04 and sub06 share one FILE formula and both must appear.
            add(anchor, force=True)

    for voice_pool in pool_options:
        music_pool_options = (voice_pool,) if tie_voice_music_pooling else pool_options
        for music_pool in music_pool_options:
            for combine in combine_options:
                for voice_exponent in exponent_options:
                    for music_exponent in exponent_options:
                        for weight in weight_options:
                            active_mixture = mixture_options if weight > 0.0 else (mixture_options[0],)
                            for mixture_pool in active_mixture:
                                name = (
                                    f"v{voice_pool.name}-m{music_pool.name}-{combine}"
                                    f"-ve{voice_exponent:g}-me{music_exponent:g}"
                                    f"-mix{weight:g}"
                                )
                                if weight > 0.0:
                                    name += f"-{mixture_pool.name}"
                                add(
                                    FormulaSpec(
                                        name=name,
                                        voice_pooling=voice_pool,
                                        music_pooling=music_pool,
                                        mixture_pooling=mixture_pool,
                                        combine=combine,
                                        voice_exponent=voice_exponent,
                                        music_exponent=music_exponent,
                                        mixture_weight=weight,
                                    )
                                )
    return grid


# ---------------------------------------------------------------------------
# 7. Robust evaluation and ranking
# ---------------------------------------------------------------------------

def evaluate_grid_entry(
    records: Sequence[IntermediateRecord],
    formula: FormulaSpec,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    blend_weight: float = 0.5,
) -> dict:
    """Score every record with one formula and summarise robustly.

    Reported numbers:

    * overall EER/AUC on the whole subset (in-sample, optimistic);
    * grouped bootstrap CI resampling content groups, because paired real/fake
      items sharing a source recording are correlated;
    * leave-one-generator-family-out EER against all real files, so one easy
      family cannot hide a hard one;
    * worst_family_eer, the maximum of those, used for selection.

    blend_eer = (1-w)*worst_family + w*overall is offered as a less extreme
    robust summary; selection defaults to worst_family.
    """
    if not records:
        raise ValueError("no records to evaluate")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be in [0, 1]")
    scores = np.asarray([evaluate_formula(record, formula) for record in records], dtype=np.float64)
    labels = np.asarray([int(record.label) for record in records], dtype=np.int64)
    groups = [record.group for record in records]
    families = [record.generator_family for record in records]

    overall = summarize(labels, scores, subset="all")
    family_reports = leave_one_generator_family_out(labels, scores, families)
    family_eers = [report.eer for report in family_reports if report.eer is not None]
    worst_family = float(max(family_eers)) if family_eers else None
    overall_eer = overall.eer
    if worst_family is not None and overall_eer is not None:
        blend = float((1.0 - blend_weight) * worst_family + blend_weight * overall_eer)
    else:
        blend = None
    interval = grouped_bootstrap_eer_ci(
        labels, scores, groups, n_resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    return {
        "formula": formula.as_dict(),
        "formula_key": formula.key,
        "overall": overall.as_dict(),
        "worst_family_eer": worst_family,
        "worst_family": next(
            (
                report.subset
                for report in family_reports
                if report.eer is not None and worst_family is not None and report.eer == worst_family
            ),
            None,
        ),
        "blend_eer": blend,
        "blend_weight": float(blend_weight),
        "leave_one_family_out": [report.as_dict() for report in family_reports],
        "bootstrap_eer_ci": interval,
        "scores": [
            {
                "sample_id": record.sample_id,
                "label": int(record.label),
                "group": record.group,
                "generator_family": record.generator_family,
                "score": float(value),
            }
            for record, value in zip(records, scores)
        ],
    }


def ranking_value(entry: Mapping, selection_metric: str = "worst_family") -> float:
    """Primary sort value for one grid entry (lower is better)."""
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    key = {"worst_family": "worst_family_eer", "blend": "blend_eer", "overall": None}[selection_metric]
    value = entry["overall"]["eer"] if key is None else entry.get(key)
    if value is None:
        # An undefined metric must never win; push it to the end deterministically.
        return float("inf")
    return float(value)


def rank_formulas(entries: Sequence[Mapping], selection_metric: str = "worst_family") -> list[dict]:
    """Rank grid entries deterministically.

    Sort keys, in order: the selection metric, then overall EER, then the upper
    bootstrap bound, then the formula key. The trailing key makes the order
    total, so equal-scoring formulas never swap between runs.
    """
    ordered = sorted(
        entries,
        key=lambda entry: (
            ranking_value(entry, selection_metric),
            float(entry["overall"]["eer"]) if entry["overall"]["eer"] is not None else float("inf"),
            float(entry["bootstrap_eer_ci"]["upper"])
            if entry["bootstrap_eer_ci"].get("upper") is not None
            else float("inf"),
            str(entry["formula_key"]),
        ),
    )
    ranked = []
    for position, entry in enumerate(ordered, start=1):
        item = dict(entry)
        item["rank"] = position
        item["selection_metric"] = selection_metric
        item["selection_value"] = ranking_value(entry, selection_metric)
        ranked.append(item)
    return ranked


def sweep(
    records: Sequence[IntermediateRecord],
    grid: Sequence[FormulaSpec] | None = None,
    selection_metric: str = "worst_family",
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    blend_weight: float = 0.5,
) -> dict:
    """Evaluate a formula grid over cached records and rank the results."""
    if not records:
        raise ValueError("no records to sweep")
    labels = {int(record.label) for record in records}
    if len(labels) < 2:
        raise ValueError("records contain a single class; EER is undefined")
    formulas = list(grid if grid is not None else build_grid())
    ordered_records = sorted(records, key=lambda item: item.sample_id)
    entries = [
        evaluate_grid_entry(
            ordered_records,
            formula,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            blend_weight=blend_weight,
        )
        for formula in formulas
    ]
    ranked = rank_formulas(entries, selection_metric)
    anchors = {
        entry["formula"]["anchor"]: {
            "name": entry["formula"]["name"],
            "rank": entry["rank"],
            "overall_eer": entry["overall"]["eer"],
            "worst_family_eer": entry["worst_family_eer"],
        }
        for entry in ranked
        if entry["formula"]["anchor"]
    }
    return {
        "n_records": len(ordered_records),
        "n_real": sum(1 for record in ordered_records if record.label == 0),
        "n_fake": sum(1 for record in ordered_records if record.label == 1),
        "n_groups": len({record.group for record in ordered_records}),
        "n_formulas": len(formulas),
        "selection_metric": selection_metric,
        "blend_weight": float(blend_weight),
        "bootstrap": {"n_resamples": int(bootstrap_resamples), "seed": int(bootstrap_seed)},
        "domain_limitation": DOMAIN_LIMITATION,
        "score_normalisation": "none (each file scored from its own record only)",
        "anchors": anchors,
        "best": ranked[0] if ranked else None,
        "ranking": ranked,
    }


# ---------------------------------------------------------------------------
# 8. Flat CSV outputs
# ---------------------------------------------------------------------------

def write_scores_csv(path, report: Mapping, top_n: int | None = None) -> Path:
    """Write per-file scores in long format for the top ranked formulas."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranking = list(report.get("ranking", ()))
    if top_n is not None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        ranking = ranking[:top_n]
    fieldnames = [
        "rank", "formula", "anchor", "formula_key", "sample_id", "label",
        "content_group", "generator_family", "score",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ranking:
            for item in entry["scores"]:
                writer.writerow(
                    {
                        "rank": entry["rank"],
                        "formula": entry["formula"]["name"],
                        "anchor": entry["formula"]["anchor"],
                        "formula_key": entry["formula_key"],
                        "sample_id": item["sample_id"],
                        "label": item["label"],
                        "content_group": item["group"],
                        "generator_family": item["generator_family"],
                        "score": item["score"],
                    }
                )
    return target


def write_ranking_csv(path, report: Mapping) -> Path:
    """Write one row per formula with the metrics used for selection."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "formula", "anchor", "formula_key", "selection_metric", "selection_value",
        "overall_eer", "overall_auc", "worst_family_eer", "worst_family", "blend_eer",
        "bootstrap_lower", "bootstrap_upper", "tie_rate", "voice_pooling", "music_pooling",
        "mixture_pooling", "combine", "voice_exponent", "music_exponent", "mixture_weight",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report.get("ranking", ()):
            formula = entry["formula"]
            interval = entry["bootstrap_eer_ci"]
            writer.writerow(
                {
                    "rank": entry["rank"],
                    "formula": formula["name"],
                    "anchor": formula["anchor"],
                    "formula_key": entry["formula_key"],
                    "selection_metric": entry["selection_metric"],
                    "selection_value": entry["selection_value"],
                    "overall_eer": entry["overall"]["eer"],
                    "overall_auc": entry["overall"]["roc_auc"],
                    "worst_family_eer": entry["worst_family_eer"],
                    "worst_family": entry["worst_family"],
                    "blend_eer": entry["blend_eer"],
                    "bootstrap_lower": interval.get("lower"),
                    "bootstrap_upper": interval.get("upper"),
                    "tie_rate": entry["overall"]["tie_rate"],
                    "voice_pooling": formula["voice_pooling"]["name"],
                    "music_pooling": formula["music_pooling"]["name"],
                    "mixture_pooling": formula["mixture_pooling"]["name"],
                    "combine": formula["combine"],
                    "voice_exponent": formula["voice_exponent"],
                    "music_exponent": formula["music_exponent"],
                    "mixture_weight": formula["mixture_weight"],
                }
            )
    return target


# ---------------------------------------------------------------------------
# 9. Extraction (lazy heavy imports; nothing above this line loads a model)
# ---------------------------------------------------------------------------

def _require(module_name: str, hint: str):
    """Import a soft dependency, converting failure into a clear error."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ExtractionUnavailableError(
            f"{module_name} is required for extraction but is not importable ({exc}). {hint}"
        ) from exc


def calculate_rms(audio) -> float:
    """Baseline calculate_rms: float64 mean square then sqrt."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


class IntermediateExtractor:
    """Runs PANNs, HTDemucs and DF-Arena once per file and returns records.

    Models load lazily on first use so that importing this module, building
    grids, and running the whole sweep stay dependency-free. Only scalars and
    short per-segment arrays are kept; separated stems are discarded as soon as
    their segment scores are computed, so the cache stays small.
    """

    def __init__(self, model_dir, config: ExtractionConfig | None = None) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.config = config or ExtractionConfig()
        if not (self.model_dir / "df_arena_1b" / "config.json").is_file():
            raise ExtractionUnavailableError(
                f"DF-Arena 1B checkpoint not found under {self.model_dir}"
            )
        self._torch = None
        self._df_model = None
        self._fake_index = None
        self._demucs = None
        self._panns = None

    # -- lazy loaders ------------------------------------------------------

    def torch(self):
        if self._torch is None:
            self._torch = _require("torch", "Install torch to run extraction.")
        return self._torch

    def df_model(self):
        """Load DF-Arena 1B exactly like baseline load_df_arena_model."""
        if self._df_model is None:
            import sys

            torch = self.torch()
            if str(self.model_dir) not in sys.path:
                sys.path.insert(0, str(self.model_dir))
            module = _require(
                "df_arena_1b.modeling_antispoofing",
                f"Ensure {self.model_dir} contains the baseline model package.",
            )
            checkpoint_dir = self.model_dir / "df_arena_1b"
            previous = Path.cwd()
            os.chdir(checkpoint_dir)
            try:
                model = module.DF_Arena_1B_Antispoofing.from_pretrained(
                    str(checkpoint_dir), local_files_only=True, low_cpu_mem_usage=True
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a load error
                raise ExtractionUnavailableError(f"failed to load DF-Arena 1B: {exc}") from exc
            finally:
                os.chdir(previous)
            self._df_model = model.to(torch.device(self.config.device)).eval()
            self._fake_index = int(self._df_model.config.label2id["spoof"])
        return self._df_model, self._fake_index

    def demucs_model(self):
        """Load HTDemucs exactly like baseline load_htdemucs_model."""
        if self._demucs is None:
            torch = self.torch()
            pretrained = _require("demucs.pretrained", "Install the inference extra.")
            original_load = torch.load

            def load_trusted_checkpoint(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = load_trusted_checkpoint
            try:
                model = pretrained.get_model("htdemucs", repo=self.model_dir / "htdemucs")
            except Exception as exc:  # noqa: BLE001 - surfaced as a load error
                raise ExtractionUnavailableError(f"failed to load HTDemucs: {exc}") from exc
            finally:
                torch.load = original_load
            self._demucs = model.cpu().eval()
        return self._demucs

    def panns_model(self):
        """Load PANNs plus label groups exactly like baseline load_panns_model."""
        if self._panns is None:
            import shutil

            panns_dir = self.model_dir / "panns"
            source = panns_dir / "class_labels_indices.csv"
            target = Path.home() / "panns_data" / "class_labels_indices.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            module = _require("panns_inference", "Install panns-inference.")
            model = module.AudioTagging(
                checkpoint_path=str(panns_dir / self.config.panns_checkpoint),
                device=self.config.device,
            )
            groups = json.loads((panns_dir / "component_labels.json").read_text(encoding="utf-8"))
            label_to_index = {label: index for index, label in enumerate(module.labels)}
            voice_indices = [label_to_index[label] for label in groups["voice"]]
            music_indices = [label_to_index[label] for label in groups["music"]]
            self._panns = (model, voice_indices, music_indices)
        return self._panns

    # -- per-file computation ---------------------------------------------

    def load_audio(self, audio_path) -> np.ndarray:
        """Baseline load_audio, plus the optional research duration crop."""
        librosa = _require("librosa", "Install the inference extra.")
        audio, _ = librosa.load(
            str(audio_path), sr=self.config.audio_sample_rate, mono=True, dtype=np.float32
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ExtractionFailure(f"invalid audio: {audio_path}")
        limit = self.config.max_audio_samples
        if limit is not None:
            audio = crop_or_pad(audio, limit, self.config.crop_mode)
        return audio

    def presence(self, audio) -> tuple[float, float]:
        """Baseline predict_presence: per-group max over resampled segments."""
        librosa = _require("librosa", "Install the inference extra.")
        model, voice_indices, music_indices = self.panns_model()
        segments = []
        for start in baseline_segment_starts(int(audio.size), self.config.segment_samples):
            segment = baseline_extract_segment(audio, start, self.config.segment_samples)
            segments.append(
                librosa.resample(
                    segment,
                    orig_sr=self.config.audio_sample_rate,
                    target_sr=self.config.panns_sample_rate,
                    res_type=self.config.resample_type,
                ).astype(np.float32)
            )
        predictions, _ = model.inference(np.stack(segments))
        return (
            float(predictions[:, voice_indices].max()),
            float(predictions[:, music_indices].max()),
        )

    def separate(self, audio_path) -> tuple[np.ndarray, np.ndarray]:
        """Baseline separate_voice_and_music, including the degenerate branch."""
        torch = self.torch()
        torchaudio = _require("torchaudio", "Install torchaudio to resample stems.")
        apply_model = _require("demucs.apply", "Install demucs.").apply_model
        load_track = _require("demucs.separate", "Install demucs.").load_track
        model = self.demucs_model()
        waveform = load_track(Path(audio_path), model.audio_channels, model.samplerate).float()
        mono = waveform.mean(0)
        mean = mono.mean()
        std = mono.std()
        if float(std) < 1e-8:
            length = round(waveform.shape[-1] * self.config.audio_sample_rate / model.samplerate)
            silence = np.zeros(max(1, length), dtype=np.float32)
            return silence, silence.copy()
        normalized = (waveform - mean) / std
        with torch.inference_mode():
            sources = apply_model(
                model,
                normalized[None],
                device=torch.device(self.config.device),
                shifts=self.config.demucs_shifts,
                split=self.config.demucs_split,
                overlap=self.config.demucs_overlap,
                progress=False,
            )[0]
        sources = sources * std + mean
        vocal_index = model.sources.index("vocals")
        voice = sources[vocal_index].mean(0, keepdim=True)
        music_sources = [
            sources[index]
            for index, name in enumerate(model.sources)
            if name != "vocals"
        ]
        music = torch.stack(music_sources).sum(0).mean(0, keepdim=True)
        voice = torchaudio.functional.resample(
            voice, model.samplerate, self.config.audio_sample_rate
        )[0]
        music = torchaudio.functional.resample(
            music, model.samplerate, self.config.audio_sample_rate
        )[0]
        return (
            voice.cpu().numpy().astype(np.float32),
            music.cpu().numpy().astype(np.float32),
        )

    def stream_scores(self, audio) -> StreamScores:
        """Per-segment DF-Arena spoof probabilities, silence branch preserved."""
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return StreamScores.silence()
        rms = calculate_rms(values)
        if rms < self.config.silence_rms:
            return StreamScores.silence(rms=rms, n_samples=int(values.size))
        torch = self.torch()
        model, fake_index = self.df_model()
        device = torch.device(self.config.device)
        segments: list[float] = []
        for start in baseline_segment_starts(int(values.size), self.config.segment_samples):
            segment = baseline_extract_segment(values, start, self.config.segment_samples)
            tensor = torch.from_numpy(segment).to(device)
            with torch.inference_mode():
                logits = model(input_values=tensor)["logits"]
                probabilities = torch.softmax(logits.float(), dim=-1)
            segments.append(float(probabilities[0, fake_index]))
        return StreamScores(segments=tuple(segments), silent=False, rms=rms, n_samples=int(values.size))

    def extract(self, row: ManifestRow, audio_sha256: str | None = None) -> IntermediateRecord:
        """Extract one file's complete intermediate record."""
        started = time.perf_counter()
        path = Path(row.audio_path)
        if not path.is_file():
            raise ExtractionFailure(f"audio file not found: {path}")
        digest = audio_sha256 or file_digest(path)
        audio = self.load_audio(path)
        presence_failed = False
        try:
            voice_present, music_present = self.presence(audio)
        except Exception as exc:  # noqa: BLE001 - baseline falls back to neutral
            # Baseline predict_presence_for_all_files logs and uses (0.5, 0.5).
            presence_failed = True
            voice_present = music_present = NEUTRAL_PRESENCE
            del exc
        voice_audio, music_audio = self.separate(path)
        return IntermediateRecord(
            sample_id=row.sample_id,
            audio_sha256=digest,
            config_fingerprint=self.config.fingerprint,
            label=int(row.label),
            group=row.group,
            generator_family=row.generator_family,
            license_id=row.license_id,
            transform=row.transform,
            voice_present=float(voice_present),
            music_present=float(music_present),
            voice=self.stream_scores(voice_audio),
            music=self.stream_scores(music_audio),
            mixture=self.stream_scores(audio),
            duration_seconds=float(audio.size) / float(self.config.audio_sample_rate),
            presence_failed=presence_failed,
            seconds=time.perf_counter() - started,
        )


def extract_records(
    rows: Sequence[ManifestRow],
    extractor_factory,
    store: IntermediateStore,
    config: ExtractionConfig,
    progress=None,
) -> dict:
    """Extract missing records for rows, reusing cached ones.

    The extractor is built lazily through extractor_factory so a fully cached
    run never loads a model. Each new record is appended immediately, so an
    interrupted run resumes from where it stopped.
    """
    if not rows:
        raise ValueError("no rows to extract")
    extractor = None
    records: list[IntermediateRecord] = []
    failures: list[dict] = []
    n_cached = 0
    n_extracted = 0
    for index, row in enumerate(rows, start=1):
        try:
            path = Path(row.audio_path)
            if not path.is_file():
                raise ExtractionFailure(f"audio file not found: {path}")
            digest = file_digest(path)
            cached = store.get(config.cache_key(digest))
            if cached is not None:
                records.append(cached)
                n_cached += 1
            else:
                if extractor is None:
                    extractor = extractor_factory()
                record = extractor.extract(row, audio_sha256=digest)
                store.append(record)
                records.append(record)
                n_extracted += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a run
            failures.append(
                {"sample_id": row.sample_id, "error_type": type(exc).__name__, "message": str(exc)}
            )
        if progress is not None:
            progress(index, len(rows))
    return {
        "records": records,
        "failures": failures,
        "n_cached": n_cached,
        "n_extracted": n_extracted,
        "corrupt_cache_lines": store.corrupt_lines,
    }
