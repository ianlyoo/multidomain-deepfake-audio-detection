"""Paired benchmark harness for music deepfake detectors.

The harness compares the official baseline music score against SONICS
SpecTTTra checkpoints on a paired manifest, scoring every file independently
and reusing the project's exact competition metric implementation.

SONICS preprocessing is reproduced from the pinned upstream revision
9156ffad151f797c71556923c4a02fa01fa8fc91 of https://github.com/awsaf49/sonics
(installed as sonics==0.1.0; see the distribution's direct_url.json):

* sonics/utils/dataset.py AudioDataset.crop_or_pad -- evaluation runs with
  random_sampling=False, which zero-pads short audio at the end and crops
  long audio starting at int((audio_len - max_len) / 4 * 3).
* sonics/utils/dataset.py AudioDataset.__getitem__ -- the fixed-length
  waveform is divided by np.maximum(np.std(audio), 1e-6) when normalize=="std".
* sonics/utils/config.py dict2cfg -- max_len = max_time * sample_rate, so the
  checkpoint config alone fixes input length (5 s -> 80000, 120 s -> 1920000).
* sonics/layers/feature.py FeatureExtractor -- mel-spectrogram, dB conversion
  and mean/std spectrogram normalisation live inside the checkpoint, so the
  harness must feed raw fixed-length waveforms only.

The authors' hosted demo (app.py of the Space
awsaf49/sonics-fake-song-detection) instead loads with
librosa.load(path, sr=16000) and takes the middle fixed-length chunk without
waveform normalisation. Both crop modes are implemented; see CROP_MODES. The
training/eval loader is the default because the published checkpoint metrics
were measured with it.

Resampling note: AudioDataset calls librosa.load(path, sr=None) because the
SONICS corpus is already 16 kHz. Arbitrary corpora are not, so the harness
resamples to cfg.audio.sample_rate (16 kHz). That matches the authors' demo
and is the only reading consistent with the checkpoint mel config (f_max=8000).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import minimum_filter1d

from .metrics import official_eer

SONICS_REPOSITORY = "https://github.com/awsaf49/sonics"
SONICS_REVISION = "9156ffad151f797c71556923c4a02fa01fa8fc91"
SONICS_SAMPLE_RATE = 16_000

CROP_MODES = ("dataset_eval", "demo_middle")
NORMALIZE_MODES = ("std", "minmax", "none")
INPUT_KINDS = ("mixture", "music_stem")


class DetectorUnavailableError(RuntimeError):
    """Raised when a detector's dependencies or checkpoints are missing."""


class ScoringFailure(RuntimeError):
    """Raised when a single file cannot be scored by a detector."""


# ---------------------------------------------------------------------------
# 1. SONICS fixed-duration preprocessing (pinned reproduction)
# ---------------------------------------------------------------------------

def crop_or_pad(audio, max_len: int, crop_mode: str = "dataset_eval") -> np.ndarray:
    """Return a deterministic fixed-length view of the waveform.

    dataset_eval mirrors AudioDataset.crop_or_pad(random_sampling=False) and
    demo_middle mirrors the Hugging Face Space middle-chunk crop.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}")
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio is empty")
    audio_len = int(waveform.shape[0])
    if crop_mode == "dataset_eval":
        if audio_len < max_len:
            return np.pad(waveform, (0, max_len - audio_len), mode="constant")
        if audio_len > max_len:
            index = int((audio_len - max_len) / 4 * 3)
            return waveform[index : index + max_len].copy()
        return waveform.copy()
    total_chunks = audio_len // max_len
    start = (total_chunks // 2) * max_len
    chunk = waveform[start : start + max_len]
    if chunk.shape[0] < max_len:
        chunk = np.pad(chunk, (0, max_len - chunk.shape[0]), mode="constant")
    return chunk.copy()


def normalize_waveform(audio, normalize: str = "std") -> np.ndarray:
    """Apply AudioDataset.__getitem__ waveform normalisation."""
    if normalize not in NORMALIZE_MODES:
        raise ValueError(f"normalize must be one of {NORMALIZE_MODES}")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if normalize == "std":
        waveform /= np.maximum(np.std(waveform), 1e-6)
    elif normalize == "minmax":
        waveform -= np.min(waveform)
        waveform /= np.maximum(np.max(waveform), 1e-6)
    return waveform


def prepare_sonics_waveform(
    audio,
    max_len: int,
    crop_mode: str = "dataset_eval",
    normalize: str = "std",
) -> np.ndarray:
    """Crop/pad then normalise a waveform exactly like the pinned SONICS loader."""
    return normalize_waveform(crop_or_pad(audio, max_len, crop_mode), normalize)


# ---------------------------------------------------------------------------
# 2. Manifest loading with configurable column names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestColumns:
    """Configurable manifest column names (CLI-overridable)."""

    rel_path: str = "rel_path"
    audio_path: str = "audio_path"
    label: str = "label"
    group: str = "content_group"
    generator_family: str = "generator_family"
    license_id: str = "license"
    transform: str = "transform"


@dataclass(frozen=True)
class ManifestRow:
    """One paired-manifest entry resolved to an absolute audio path."""

    sample_id: str
    audio_path: Path
    label: int
    group: str
    generator_family: str
    license_id: str
    transform: str


REAL_LABELS = frozenset({"real", "bonafide", "genuine", "0"})
FAKE_LABELS = frozenset({"fake", "spoof", "synthetic", "1"})


def parse_label(value) -> int:
    """Map a manifest label token to 1 (fake) or 0 (real)."""
    token = str(value).strip().lower()
    if token in FAKE_LABELS:
        return 1
    if token in REAL_LABELS:
        return 0
    raise ValueError(f"unrecognised label: {value!r}")


def load_manifest(
    path,
    data_root=None,
    columns: ManifestColumns | None = None,
) -> list[ManifestRow]:
    """Read a paired manifest CSV into resolved rows.

    Either the rel_path column (resolved against data_root) or the audio_path
    column must be present. Missing optional metadata falls back to "unknown".
    """
    import csv

    spec = columns or ManifestColumns()
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    has_rel = spec.rel_path in fieldnames
    has_abs = spec.audio_path in fieldnames
    if not (has_rel or has_abs):
        raise ValueError(
            f"manifest must contain {spec.rel_path!r} or {spec.audio_path!r}; "
            f"found {fieldnames}"
        )
    if spec.label not in fieldnames:
        raise ValueError(f"manifest must contain label column {spec.label!r}")
    root = Path(data_root) if data_root is not None else manifest_path.parent

    parsed: list[ManifestRow] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        relative = ""
        if has_rel:
            relative = (row.get(spec.rel_path) or "").strip().replace("\\", "/")
        absolute = (row.get(spec.audio_path) or "").strip() if has_abs else ""
        if absolute:
            audio_path = Path(absolute)
        elif relative:
            if ".." in Path(relative).parts:
                raise ValueError(f"row {number} has an unsafe rel_path")
            audio_path = root / relative
        else:
            raise ValueError(f"row {number} has no audio path")
        sample_id = relative or str(audio_path)
        if sample_id in seen:
            raise ValueError(f"row {number} duplicates sample {sample_id!r}")
        seen.add(sample_id)
        try:
            label = parse_label(row[spec.label])
        except ValueError as exc:
            raise ValueError(f"row {number}: {exc}") from exc
        parsed.append(
            ManifestRow(
                sample_id=sample_id,
                audio_path=audio_path,
                label=label,
                group=(row.get(spec.group) or "").strip() or "unknown",
                generator_family=(row.get(spec.generator_family) or "").strip() or "unknown",
                license_id=(row.get(spec.license_id) or "").strip() or "unknown",
                transform=(row.get(spec.transform) or "").strip() or "unknown",
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# 3. Grouped metrics, bootstrap CIs and leave-one-generator-family-out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSummary:
    """Primary metrics for one detector over one evaluation subset."""

    subset: str
    n_files: int
    n_real: int
    n_fake: int
    eer: float | None
    roc_auc: float | None
    tie_rate: float
    distinct_score_ratio: float
    note: str = ""

    def as_dict(self) -> dict:
        return dict(vars(self))


def tie_statistics(scores) -> tuple[float, float]:
    """Return (tie_rate, distinct_score_ratio) for a score vector.

    tie_rate is the fraction of files whose score is shared with another file,
    which is the practical cause of degenerate EER on saturated detectors.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    unique, counts = np.unique(values, return_counts=True)
    tied = int(counts[counts > 1].sum())
    return float(tied / values.size), float(unique.size / values.size)


def _eer_or_none(labels, scores) -> tuple[float | None, float | None, str]:
    """Compute EER/ROC-AUC, returning None when a class is missing."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return None, None, "empty subset"
    if np.unique(y_true).size < 2:
        return None, None, "single-class subset"
    eer = official_eer(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return eer, auc, ""


def summarize(labels, scores, subset: str = "all") -> MetricSummary:
    """Summarise one subset using the project's exact EER implementation."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have the same shape")
    eer, auc, note = _eer_or_none(y_true, y_score)
    tie_rate, distinct_ratio = tie_statistics(y_score)
    return MetricSummary(
        subset=subset,
        n_files=int(y_true.size),
        n_real=int((y_true == 0).sum()),
        n_fake=int((y_true == 1).sum()),
        eer=eer,
        roc_auc=auc,
        tie_rate=tie_rate,
        distinct_score_ratio=distinct_ratio,
        note=note,
    )


def grouped_bootstrap_eer_ci(
    labels,
    scores,
    groups,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap an EER confidence interval by resampling whole groups.

    Paired real/fake items that share an original recording are correlated, so
    the cluster bootstrap resamples content groups (not files) to avoid the
    optimistic intervals a naive file-level bootstrap would produce.
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    keys = np.asarray([str(g) for g in groups]).reshape(-1)
    if not (y_true.shape == y_score.shape == keys.shape):
        raise ValueError("labels, scores and groups must have the same shape")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    unique_groups = np.unique(keys)
    index_by_group = {group: np.flatnonzero(keys == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    degenerate = 0
    for _ in range(n_resamples):
        picked = generator.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([index_by_group[group] for group in picked])
        sample_true = y_true[indices]
        if np.unique(sample_true).size < 2:
            degenerate += 1
            continue
        estimates.append(official_eer(sample_true, y_score[indices]))
    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0
    if not estimates:
        return {
            "n_groups": int(unique_groups.size),
            "n_resamples": int(n_resamples),
            "n_effective": 0,
            "n_degenerate": int(degenerate),
            "confidence": float(confidence),
            "seed": int(seed),
            "lower": None,
            "upper": None,
            "median": None,
        }
    array = np.asarray(estimates, dtype=np.float64)
    return {
        "n_groups": int(unique_groups.size),
        "n_resamples": int(n_resamples),
        "n_effective": int(array.size),
        "n_degenerate": int(degenerate),
        "confidence": float(confidence),
        "seed": int(seed),
        "lower": float(np.percentile(array, lower_q)),
        "upper": float(np.percentile(array, upper_q)),
        "median": float(np.median(array)),
    }


def leave_one_generator_family_out(labels, scores, families, real_family="unknown") -> list[MetricSummary]:
    """Report EER per held-out generator family against all real files.

    Each report keeps every real file and only the fake files of one family, so
    a single family cannot hide behind an easier one in the pooled number.
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    keys = np.asarray([str(f) for f in families]).reshape(-1)
    if not (y_true.shape == y_score.shape == keys.shape):
        raise ValueError("labels, scores and families must have the same shape")
    real_mask = y_true == 0
    reports: list[MetricSummary] = []
    fake_families = sorted({key for key, label in zip(keys, y_true) if label == 1})
    for family in fake_families:
        mask = real_mask | ((y_true == 1) & (keys == family))
        reports.append(summarize(y_true[mask], y_score[mask], subset=f"family={family}"))
    return reports


def per_group_summaries(labels, scores, groups) -> list[MetricSummary]:
    """Summarise metrics within each content group."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    keys = np.asarray([str(g) for g in groups]).reshape(-1)
    reports: list[MetricSummary] = []
    for group in sorted(set(keys.tolist())):
        mask = keys == group
        reports.append(summarize(y_true[mask], y_score[mask], subset=f"group={group}"))
    return reports


# ---------------------------------------------------------------------------
# 4. Score cache keyed by file hash + detector config
# ---------------------------------------------------------------------------

def file_digest(path, chunk_bytes: int = 1024 * 1024) -> str:
    """SHA-256 of a file, matching the manifest module's hashing."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: Mapping) -> str:
    """Stable digest of a detector configuration mapping."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ScoreCache:
    """JSON-backed per-file score cache keyed by file hash + detector config.

    The cache stores only values a detector already produced, so a hit returns
    the same float the detector would return. Nothing about cache state is
    visible to a detector, therefore predictions cannot depend on it.
    """

    def __init__(self, path=None, enabled: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = bool(enabled and self.path is not None)
        self._entries: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            if isinstance(raw, dict):
                for key, value in raw.get("scores", {}).items():
                    try:
                        self._entries[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue

    @staticmethod
    def make_key(file_hash: str, detector_key: str) -> str:
        return f"{detector_key}:{file_hash}"

    def get(self, key: str) -> float | None:
        if not self.enabled:
            return None
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        return None

    def put(self, key: str, score: float) -> None:
        if self.enabled:
            self._entries[key] = float(score)

    def flush(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "scores": dict(sorted(self._entries.items()))}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Detectors
# ---------------------------------------------------------------------------

@dataclass
class DetectorSpec:
    """Identity and configuration of one scored detector variant."""

    name: str
    kind: str
    input_kind: str = "mixture"
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_kind not in INPUT_KINDS:
            raise ValueError(f"input_kind must be one of {INPUT_KINDS}")

    @property
    def key(self) -> str:
        """Cache key component: name, input kind and config fingerprint."""
        payload = dict(self.config)
        payload["__kind__"] = self.kind
        payload["__input__"] = self.input_kind
        return f"{self.name}|{self.input_kind}|{config_fingerprint(payload)}"


class Detector:
    """Scores one audio file at a time; higher score means more likely fake."""

    spec: DetectorSpec

    def score(self, audio_path) -> float:
        raise NotImplementedError

    def close(self) -> None:
        return None


def _require(module_name: str, hint: str):
    """Import a soft dependency, converting failure into a clear error."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DetectorUnavailableError(
            f"{module_name} is required for this detector but is not importable "
            f"({exc}). {hint}"
        ) from exc


def load_audio_16k(audio_path, sample_rate: int = SONICS_SAMPLE_RATE) -> np.ndarray:
    """Load mono float32 audio at the detector sample rate."""
    librosa = _require("librosa", "Install the inference extra: uv sync --extra inference")
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ScoringFailure(f"invalid audio: {audio_path}")
    return audio


class SonicsDetector(Detector):
    """SONICS SpecTTTra checkpoint scored with the authors' fixed-duration input.

    The checkpoint owns mel extraction and spectrogram normalisation, so this
    detector only reproduces waveform-level preprocessing and calls the model
    once per file with a single fixed-length window.
    """

    def __init__(
        self,
        checkpoint_dir,
        name=None,
        device: str = "cpu",
        crop_mode: str = "dataset_eval",
        normalize=None,
        input_kind: str = "mixture",
        stem_provider=None,
    ) -> None:
        checkpoint_path = Path(checkpoint_dir)
        if not (checkpoint_path / "config.json").is_file():
            raise DetectorUnavailableError(
                f"SONICS checkpoint config not found: {checkpoint_path / 'config.json'}"
            )
        if not (checkpoint_path / "pytorch_model.bin").is_file():
            raise DetectorUnavailableError(
                f"SONICS weights not found: {checkpoint_path / 'pytorch_model.bin'}"
            )
        self._torch = _require("torch", "Install torch to run SONICS checkpoints.")
        sonics = _require(
            "sonics",
            "Install the pinned revision: "
            f"uv pip install git+{SONICS_REPOSITORY}@{SONICS_REVISION}",
        )
        raw_config = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
        audio_config = raw_config.get("audio", {})
        self.sample_rate = int(audio_config.get("sample_rate", SONICS_SAMPLE_RATE))
        self.max_time = float(audio_config.get("max_time"))
        # sonics.utils.config.dict2cfg recomputes max_len from max_time; mirror it.
        self.max_len = int(self.max_time * self.sample_rate)
        if normalize is None:
            normalize = resolve_normalize_mode(audio_config)
        self.crop_mode = crop_mode
        self.normalize = normalize
        self.stem_provider = stem_provider
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")
        if normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}")

        try:
            model = sonics.HFAudioClassifier.from_pretrained(str(checkpoint_path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(
                f"failed to load SONICS checkpoint {checkpoint_path}: {exc}"
            ) from exc
        self.device = self._torch.device(device)
        self.model = model.to(self.device).eval()

        label = name or f"sonics-{raw_config.get('experiment_name', checkpoint_path.name)}"
        self.spec = DetectorSpec(
            name=label,
            kind="sonics",
            input_kind=input_kind,
            config={
                "checkpoint": checkpoint_path.name,
                "revision": SONICS_REVISION,
                "max_time": self.max_time,
                "max_len": self.max_len,
                "sample_rate": self.sample_rate,
                "crop_mode": self.crop_mode,
                "normalize": self.normalize,
            },
        )

    def prepare(self, audio) -> np.ndarray:
        """Fixed-duration waveform fed to the checkpoint."""
        return prepare_sonics_waveform(audio, self.max_len, self.crop_mode, self.normalize)

    def score(self, audio_path) -> float:
        audio = load_audio_16k(audio_path, self.sample_rate)
        if self.spec.input_kind == "music_stem":
            if self.stem_provider is None:
                raise ScoringFailure("music_stem input requires a stem provider")
            audio = self.stem_provider(audio_path)
        window = self.prepare(audio)
        torch = self._torch
        tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits.float()).reshape(-1)[0]
        return float(probability.item())


BASELINE_SEGMENT_SAMPLES = 64_600
BASELINE_SILENCE_RMS = 1e-5


def baseline_segment_starts(audio_length: int, segment_samples: int = BASELINE_SEGMENT_SAMPLES) -> list[int]:
    """Reproduce get_segment_starts from baseline/official/script.py."""
    if audio_length <= segment_samples:
        return [0]
    last_start = audio_length - segment_samples
    starts = list(range(0, last_start + 1, segment_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def baseline_extract_segment(audio, start: int, segment_samples: int = BASELINE_SEGMENT_SAMPLES) -> np.ndarray:
    """Reproduce extract_segment from baseline/official/script.py."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio is empty")
    if waveform.size < segment_samples:
        repeat_count = segment_samples // waveform.size + 1
        tiled = np.tile(waveform, repeat_count)
        return tiled[:segment_samples].astype(np.float32)
    return waveform[start : start + segment_samples].astype(np.float32, copy=False)


class DemucsStemProvider:
    """Optional HTDemucs music-stem separation matching the official baseline.

    Mirrors separate_voice_and_music from baseline/official/script.py: mean/std
    normalisation before separation, non-vocal stems summed to mono, then
    resampled to the target rate.
    """

    def __init__(self, repo_dir=None, device: str = "cpu", sample_rate: int = SONICS_SAMPLE_RATE) -> None:
        self._torch = _require("torch", "Install torch to run HTDemucs separation.")
        self._torchaudio = _require("torchaudio", "Install torchaudio to resample stems.")
        pretrained = _require(
            "demucs.pretrained",
            "Install the inference extra: uv sync --extra inference",
        )
        self._apply = _require("demucs.apply", "Install demucs to separate stems.").apply_model
        self._load_track = _require("demucs.separate", "Install demucs to load tracks.").load_track
        self.sample_rate = int(sample_rate)
        self.repo_dir = Path(repo_dir) if repo_dir is not None else None
        if self.repo_dir is not None and not self.repo_dir.is_dir():
            raise DetectorUnavailableError(f"HTDemucs repo directory not found: {self.repo_dir}")

        torch = self._torch
        original_load = torch.load

        def load_trusted_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = load_trusted_checkpoint
        try:
            model = pretrained.get_model("htdemucs", repo=self.repo_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(f"failed to load HTDemucs: {exc}") from exc
        finally:
            torch.load = original_load
        self.device = torch.device(device)
        self.model = model.cpu().eval()

    def __call__(self, audio_path) -> np.ndarray:
        torch = self._torch
        model = self.model
        waveform = self._load_track(Path(audio_path), model.audio_channels, model.samplerate).float()
        mono = waveform.mean(0)
        mean = mono.mean()
        std = mono.std()
        if float(std) < 1e-8:
            length = round(waveform.shape[-1] * self.sample_rate / model.samplerate)
            return np.zeros(max(1, length), dtype=np.float32)
        normalized = (waveform - mean) / std
        with torch.inference_mode():
            sources = self._apply(
                model,
                normalized[None],
                device=self.device,
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        sources = sources * std + mean
        music_sources = [
            sources[index]
            for index, source_name in enumerate(model.sources)
            if source_name != "vocals"
        ]
        music = torch.stack(music_sources).sum(0).mean(0, keepdim=True)
        music = self._torchaudio.functional.resample(music, model.samplerate, self.sample_rate)[0]
        return music.cpu().numpy().astype(np.float32)


class BaselineMusicDetector(Detector):
    """Official baseline music score: DF-Arena 1B applied to a music signal.

    Reproduces predict_fake from baseline/official/script.py (segment-wise max
    of the spoof probability, silence short-circuit). With a Demucs stem
    provider this is the official MUSIC_FAKE_PROB path; without one it scores
    the raw mixture, which isolates the contribution of separation.
    """

    def __init__(
        self,
        model_dir,
        device: str = "cpu",
        input_kind: str = "music_stem",
        stem_provider=None,
        name: str = "baseline-df-arena",
        sample_rate: int = SONICS_SAMPLE_RATE,
        max_audio_seconds: float | None = None,
        crop_mode: str = "dataset_eval",
    ) -> None:
        import os
        import sys

        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        self._torch = _require("torch", "Install torch to run DF-Arena 1B.")
        _require("transformers", "Install the inference extra: uv sync --extra inference")
        # Resolve to an absolute path: from_pretrained treats a relative path as a
        # Hub repo id, and the loader also chdir()s into the checkpoint directory,
        # which would invalidate any relative path anyway.
        model_root = Path(model_dir).resolve()
        checkpoint_dir = model_root / "df_arena_1b"
        if not (checkpoint_dir / "config.json").is_file():
            raise DetectorUnavailableError(
                f"DF-Arena 1B checkpoint not found: {checkpoint_dir}"
            )
        if input_kind == "music_stem" and stem_provider is None:
            raise ValueError("music_stem input requires a stem provider")
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        try:
            module = _require(
                "df_arena_1b.modeling_antispoofing",
                f"Ensure {model_root} contains the baseline model package.",
            )
        except DetectorUnavailableError:
            raise
        previous_directory = Path.cwd()
        os.chdir(checkpoint_dir)
        try:
            model = module.DF_Arena_1B_Antispoofing.from_pretrained(
                str(checkpoint_dir), local_files_only=True, low_cpu_mem_usage=True
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(f"failed to load DF-Arena 1B: {exc}") from exc
        finally:
            os.chdir(previous_directory)
        self.device = self._torch.device(device)
        self.model = model.to(self.device).eval()
        self.fake_label_index = int(self.model.config.label2id["spoof"])
        self.stem_provider = stem_provider
        self.sample_rate = int(sample_rate)
        self.max_audio_samples = (
            int(round(float(max_audio_seconds) * self.sample_rate))
            if max_audio_seconds is not None
            else None
        )
        self.crop_mode = crop_mode
        self.spec = DetectorSpec(
            name=name,
            kind="baseline_df_arena",
            input_kind=input_kind,
            config={
                "checkpoint": "df_arena_1b",
                "segment_samples": BASELINE_SEGMENT_SAMPLES,
                "silence_rms": BASELINE_SILENCE_RMS,
                "sample_rate": self.sample_rate,
                "separation": "htdemucs" if input_kind == "music_stem" else "none",
                "max_audio_seconds": max_audio_seconds,
                "crop_mode": crop_mode if max_audio_seconds is not None else None,
            },
        )

    def score(self, audio_path) -> float:
        if self.spec.input_kind == "music_stem":
            audio = np.asarray(self.stem_provider(audio_path), dtype=np.float32).reshape(-1)
            if audio.size == 0:
                raise ScoringFailure(f"empty music stem: {audio_path}")
        else:
            audio = load_audio_16k(audio_path, self.sample_rate)
        if self.max_audio_samples is not None:
            audio = crop_or_pad(audio, self.max_audio_samples, self.crop_mode)
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms < BASELINE_SILENCE_RMS:
            return 0.0
        torch = self._torch
        segment_scores: list[float] = []
        for start in baseline_segment_starts(int(audio.size)):
            segment = baseline_extract_segment(audio, start)
            tensor = torch.from_numpy(segment).to(self.device)
            with torch.inference_mode():
                logits = self.model(input_values=tensor)["logits"]
                probabilities = torch.softmax(logits.float(), dim=-1)
            segment_scores.append(float(probabilities[0, self.fake_label_index]))
        return max(segment_scores)


# ---------------------------------------------------------------------------
# lofcz AI-music fakeprint detector (pinned reference + checkpoint)
# ---------------------------------------------------------------------------
#
# Reproduces AIDetector from checkpoints/lofcz-ai-music-source at
# LOFCZ_SOURCE_REVISION (src/python/inference.py, src/python/config.yaml)
# scored with checkpoints/lofcz-ai-music-detector at LOFCZ_MODEL_REVISION.
# The local checkout history has moved on, so provenance is recorded as
# pinned revision constants plus runtime content hashes: the model bytes
# match the pinned ONNX SHA-256 and the source checkout HEAD equals the
# pinned source revision.

LOFCZ_SOURCE_REVISION = "6ba389e94a179ac90f3eb134b741ef37baa30434"
LOFCZ_MODEL_REVISION = "d2180598fed79e3f917e8050a00439982466e5c6"
LOFCZ_ONNX_SHA256 = "af7a75c6ed457bc5b6941c8bc76aa06a66d48de40db944b761ed2bebfc0fbbd3"
LOFCZ_ONNX_FILENAME = "ai_music_detector.onnx"
LOFCZ_CONFIG_FILENAME = "config.json"
LOFCZ_SAMPLE_RATE = 16_000
LOFCZ_N_FFT = 8192
LOFCZ_MAX_DURATION_SECONDS = 300
LOFCZ_MAX_SAMPLES = LOFCZ_MAX_DURATION_SECONDS * LOFCZ_SAMPLE_RATE
LOFCZ_N_FEATURES = 3585
LOFCZ_INPUT_NAME = "fakeprint"
LOFCZ_OUTPUT_NAME = "ai_probability"
LOFCZ_PROVIDERS = ["CPUExecutionProvider"]
# Pinned fakeprint parameters (src/python/config.yaml). The checkpoint
# config.json must agree exactly or construction fails closed.
LOFCZ_PREPROCESSING = {
    "sample_rate": LOFCZ_SAMPLE_RATE,
    "n_fft": LOFCZ_N_FFT,
    "freq_min": 1000,
    "freq_max": 8000,
    "hull_area": 10,
    "max_db": 5,
    "min_db": -45,
}


def lofcz_frequency_mask(
    n_fft: int = LOFCZ_N_FFT,
    sample_rate: int = LOFCZ_SAMPLE_RATE,
    freq_min: int = 1000,
    freq_max: int = 8000,
) -> np.ndarray:
    """Boolean mask over rFFT bins for the inclusive [freq_min, freq_max] band.

    Mirrors AIDetector frequency-bin construction: linspace over the one-sided
    spectrum with an inclusive comparison. With the pinned parameters this
    selects bins 512..4096, i.e. exactly 3585 features.
    """
    freq_bins = np.linspace(0.0, sample_rate / 2.0, num=(n_fft // 2) + 1)
    return (freq_bins >= freq_min) & (freq_bins <= freq_max)


def load_lofcz_audio_16k(
    audio_path,
    sample_rate: int = LOFCZ_SAMPLE_RATE,
    max_samples: int = LOFCZ_MAX_SAMPLES,
) -> np.ndarray:
    """Load mono float32 audio with the exact upstream loader (torchaudio).

    Unlike load_audio_16k (librosa), this mirrors AIDetector.load_audio so a
    resampler difference cannot shift the saturated output probabilities.
    Multi-channel audio is averaged to mono and only the first max_samples
    are kept. Short files are returned as-is; only empty, undecodable or
    non-finite input raises.
    """
    torchaudio = _require("torchaudio", "Install the inference extra: uv sync --extra inference")
    try:
        waveform, source_rate = torchaudio.load(str(audio_path))
    except Exception as exc:  # noqa: BLE001 - surfaced as a per-file failure
        raise ScoringFailure(f"could not decode audio: {audio_path} ({exc})") from exc
    waveform = waveform.float()
    if int(source_rate) != int(sample_rate):
        resampler = torchaudio.transforms.Resample(int(source_rate), int(sample_rate))
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform.reshape(-1)
    audio = np.asarray(mono.cpu().numpy(), dtype=np.float32).reshape(-1)
    audio = audio[: int(max_samples)]
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ScoringFailure(f"invalid audio: {audio_path}")
    return audio


def compute_lofcz_fakeprint(audio_16k_mono, max_samples: int = LOFCZ_MAX_SAMPLES) -> np.ndarray:
    """Fakeprint feature vector for 16 kHz mono audio on the CPU.

    Exact port of AIDetector.compute_fakeprint with the STFT pinned to the
    CPU so scores do not depend on GPU availability: truncate to the first
    max_samples, take the 8192-point power spectrogram with module defaults,
    convert with 10*log10 under a [1e-10, 1e6] clamp, average over channel
    and time, keep the inclusive 1-8 kHz band (3585 bins), subtract the
    size-10 nearest lower hull clipped at -45 dB, clip the residue to [0, 5]
    and normalise by its maximum. Short inputs are scored as-is (the centred
    STFT pads them); only empty or non-finite input raises.
    """
    torch = _require("torch", "Install torch to run the lofcz fakeprint detector.")
    torchaudio = _require("torchaudio", "Install the inference extra: uv sync --extra inference")
    waveform = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ScoringFailure("invalid audio for lofcz fakeprint")
    waveform = waveform[: int(max_samples)]
    if waveform.size == 0:
        raise ScoringFailure("invalid audio for lofcz fakeprint")
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=LOFCZ_N_FFT, power=2, normalized=False
    )
    with torch.inference_mode():
        spec = spectrogram(tensor)
    spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
    mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()
    freq_spectrum = mean_spectrum[lofcz_frequency_mask()]
    if freq_spectrum.shape != (LOFCZ_N_FEATURES,):
        raise ScoringFailure(
            f"lofcz frequency mask selected {freq_spectrum.shape}, expected ({LOFCZ_N_FEATURES},)"
        )
    hull = minimum_filter1d(freq_spectrum, size=LOFCZ_PREPROCESSING["hull_area"], mode="nearest")
    hull = np.clip(hull, LOFCZ_PREPROCESSING["min_db"], None)
    residue = np.clip(freq_spectrum - hull, 0.0, None)
    residue = np.clip(residue, 0.0, LOFCZ_PREPROCESSING["max_db"])
    return (residue / (float(np.max(residue)) + 1e-6)).astype(np.float32)


class LofczMusicDetector(Detector):
    """Pinned lofcz AI-music fakeprint detector with a CPU ONNX head.

    The feature pipeline is compute_lofcz_fakeprint; the 3585-dimensional
    vector is fed to ai_music_detector.onnx with the CPU execution provider
    (the model is 14 KB, so provider choice only affects determinism, not
    speed). Higher score means more likely fake. Every file is scored
    independently with no minimum-duration requirement.
    """

    def __init__(self, checkpoint_dir, name: str | None = None) -> None:
        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / LOFCZ_CONFIG_FILENAME
        model_path = checkpoint_path / LOFCZ_ONNX_FILENAME
        if not config_path.is_file():
            raise DetectorUnavailableError(
                f"lofcz checkpoint config not found: {config_path}"
            )
        if not model_path.is_file():
            raise DetectorUnavailableError(f"lofcz weights not found: {model_path}")
        # Import onnxruntime before torch/torchaudio: on Windows the ORT native
        # DLL (onnxruntime_pybind11_state) can fail to load when torch has
        # already pulled its bundled OpenMP/MKL DLLs into the process first.
        onnxruntime = _require("onnxruntime", "Install the inference extra: uv sync --extra inference")
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DetectorUnavailableError(
                f"could not read lofcz checkpoint config {config_path}: {exc}"
            ) from exc
        preprocessing = raw_config.get("preprocessing", {})
        for key, expected in LOFCZ_PREPROCESSING.items():
            if preprocessing.get(key) != expected:
                raise DetectorUnavailableError(
                    f"lofcz checkpoint preprocessing mismatch for {key!r}: "
                    f"expected {expected!r}, found {preprocessing.get(key)!r}"
                )
        if int(raw_config.get("input_features", LOFCZ_N_FEATURES)) != LOFCZ_N_FEATURES:
            raise DetectorUnavailableError(
                f"lofcz checkpoint input_features is {raw_config.get('input_features')!r}, "
                f"expected {LOFCZ_N_FEATURES}"
            )
        try:
            session = onnxruntime.InferenceSession(str(model_path), providers=list(LOFCZ_PROVIDERS))
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(
                f"failed to load lofcz ONNX model {model_path}: {exc}"
            ) from exc
        # torch/torchaudio are only needed for scoring; require them after the
        # ORT session exists so the ORT native module always loads first.
        _require("torch", "Install torch to run the lofcz fakeprint detector.")
        _require("torchaudio", "Install the inference extra: uv sync --extra inference")
        model_inputs = session.get_inputs()
        try:
            input_width = int(model_inputs[0].shape[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise DetectorUnavailableError(
                f"lofcz ONNX model has an unexpected input shape: {model_inputs[0].shape!r}"
            ) from exc
        if len(model_inputs) != 1 or input_width != LOFCZ_N_FEATURES:
            raise DetectorUnavailableError(
                f"lofcz ONNX model expects {[getattr(item, 'shape', None) for item in model_inputs]!r}, "
                f"expected a single [batch, {LOFCZ_N_FEATURES}] input"
            )
        self._session = session
        self._input_name = model_inputs[0].name
        self._output_name = session.get_outputs()[0].name
        self.sample_rate = LOFCZ_SAMPLE_RATE
        self.max_samples = LOFCZ_MAX_SAMPLES
        self.spec = DetectorSpec(
            name=name or "lofcz-ai-music-fakeprint",
            kind="lofcz",
            input_kind="mixture",
            config={
                "checkpoint": checkpoint_path.name,
                "model_revision": LOFCZ_MODEL_REVISION,
                "source_revision": LOFCZ_SOURCE_REVISION,
                "onnx_digest": file_digest(model_path),
                "config_digest": file_digest(config_path),
                "preprocessing": dict(LOFCZ_PREPROCESSING),
                "max_duration_seconds": LOFCZ_MAX_DURATION_SECONDS,
                "input_name": self._input_name,
                "output_name": self._output_name,
                "providers": list(LOFCZ_PROVIDERS),
                "loader": "torchaudio",
            },
        )

    def score(self, audio_path) -> float:
        audio = load_lofcz_audio_16k(audio_path, self.sample_rate, self.max_samples)
        fakeprint = compute_lofcz_fakeprint(audio, self.max_samples)
        try:
            outputs = self._session.run(
                [self._output_name], {self._input_name: fakeprint.reshape(1, -1)}
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a per-file failure
            raise ScoringFailure(f"lofcz ONNX inference failed for {audio_path}: {exc}") from exc
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability):
            raise ScoringFailure(f"lofcz detector returned a non-finite score: {probability}")
        return probability


# ---------------------------------------------------------------------------
# 6. Subset sampling for quick runs
# ---------------------------------------------------------------------------

def sample_rows(
    rows,
    max_rows=None,
    max_content_groups=None,
    balanced: bool = False,
    stratify_generators: bool = False,
    seed: int = 42,
):
    """Deterministically shrink a manifest for quick detector smoke runs.

    The paired subset is heavily fake-skewed (119 real vs 1931 fake as measured
    on the current build), so an unbalanced head-slice can easily contain zero
    real files and produce an undefined EER. Selection therefore:

    1. applies any content-group limit before row sampling;
    2. when balancing, selects one real and one fake from each chosen content
       group, so every retained comparison is genuinely paired;
    3. is seeded and sorts by sample_id, so the same arguments always select
       the same files.
    """
    selected = list(rows)
    if not selected:
        raise ValueError("no manifest rows to sample")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    generator = np.random.default_rng(seed)

    if max_content_groups is not None:
        if max_content_groups < 1:
            raise ValueError("max_content_groups must be positive")
        groups = sorted({row.group for row in selected})
        if len(groups) > max_content_groups:
            picked = set(
                generator.choice(np.asarray(groups, dtype=object), size=max_content_groups, replace=False).tolist()
            )
            selected = [row for row in selected if row.group in picked]

    if balanced:
        rows_by_group: dict[str, dict[int, list[ManifestRow]]] = {}
        for row in selected:
            rows_by_group.setdefault(row.group, {0: [], 1: []})[row.label].append(row)
        paired_groups = sorted(
            group
            for group, by_label in rows_by_group.items()
            if by_label[0] and by_label[1]
        )
        budget = len(paired_groups)
        if max_rows is not None:
            budget = min(budget, max(1, max_rows // 2))
        if budget == 0:
            raise ValueError(
                "balanced sampling requires a content group containing both "
                "a real and a fake row"
            )
        chosen_fakes: dict[str, ManifestRow] = {}
        if stratify_generators:
            candidates_by_family: dict[str, list[ManifestRow]] = {}
            for group in paired_groups:
                for fake in rows_by_group[group][1]:
                    candidates_by_family.setdefault(fake.generator_family, []).append(fake)
            for family, pool in candidates_by_family.items():
                order = generator.permutation(len(pool))
                candidates_by_family[family] = [pool[int(index)] for index in order]
            families = sorted(candidates_by_family)
            while len(chosen_fakes) < budget:
                progressed = False
                for family in families:
                    pool = candidates_by_family[family]
                    candidate = next(
                        (item for item in pool if item.group not in chosen_fakes),
                        None,
                    )
                    if candidate is None:
                        continue
                    chosen_fakes[candidate.group] = candidate
                    progressed = True
                    if len(chosen_fakes) == budget:
                        break
                if not progressed:
                    break
            if len(chosen_fakes) < budget:
                raise ValueError(
                    "generator-stratified sampling could not find enough unique paired groups"
                )
            paired_groups = sorted(chosen_fakes)
        elif len(paired_groups) > budget:
            indices = generator.choice(len(paired_groups), size=budget, replace=False)
            paired_groups = [paired_groups[int(index)] for index in sorted(indices.tolist())]

        paired_rows: list[ManifestRow] = []
        for group in paired_groups:
            by_label = rows_by_group[group]
            real_pool = sorted(by_label[0], key=lambda row: row.sample_id)
            real_index = int(generator.integers(len(real_pool))) if len(real_pool) > 1 else 0
            paired_rows.append(real_pool[real_index])
            if stratify_generators:
                paired_rows.append(chosen_fakes[group])
            else:
                fake_pool = sorted(by_label[1], key=lambda row: row.sample_id)
                fake_index = int(generator.integers(len(fake_pool))) if len(fake_pool) > 1 else 0
                paired_rows.append(fake_pool[fake_index])
        selected = paired_rows
    elif max_rows is not None:
        ordered = sorted(selected, key=lambda row: row.sample_id)
        if len(ordered) > max_rows:
            indices = generator.choice(len(ordered), size=max_rows, replace=False)
            selected = [ordered[int(index)] for index in sorted(indices.tolist())]
        else:
            selected = ordered

    selected.sort(key=lambda row: row.sample_id)
    labels = {row.label for row in selected}
    if len(labels) < 2:
        raise ValueError(
            "sampled subset has a single class; EER is undefined. "
            "Use --balanced or raise --max-rows/--max-content-groups."
        )
    return selected


# ---------------------------------------------------------------------------
# 7. Explicit fidelity decisions for SONICS preprocessing
# ---------------------------------------------------------------------------

FIDELITY_NOTES = {
    "resampling": (
        "sonics.utils.dataset.AudioDataset loads with librosa.load(sr=None), i.e. it "
        "does not resample, because every file in the SONICS corpus is already at "
        "cfg.audio.sample_rate (16 kHz). That makes sr=None and sr=16000 identical on "
        "that data but not on arbitrary corpora. Feeding a 44.1 kHz file unresampled "
        "would silently change both the analysed duration (max_len samples would cover "
        "1.8 s instead of 5 s) and the mel mapping (the checkpoint fixes f_max=8000, "
        "which is Nyquist only at 16 kHz). The harness therefore resamples to "
        "cfg.audio.sample_rate, which is what the upstream inference demo does "
        "explicitly (librosa.load(path, sr=16000)) and is the only reading that keeps "
        "the checkpoint own configuration self-consistent."
    ),
    "normalize": (
        "The checkpoint config exposes audio.normalize as a boolean, while "
        "AudioDataset takes a string mode and get_dataloader defaults it to std. "
        "The boolean is not itself the mode. The harness maps normalize=true to the "
        "loader default std (divide by max(std, 1e-6)) and normalize=false to none, "
        "and records the resolved string in the detector config so it is visible in "
        "results and in the cache key. Override with an explicit mode when comparing."
    ),
    "crop": (
        "Validation uses random_sampling=False, so cropping is deterministic: "
        "overlength audio is cropped at int((audio_len - max_len) / 4 * 3) and short "
        "audio is zero-padded on the right. The upstream demo instead takes the middle "
        "fixed-length chunk; that variant is available as crop_mode=demo_middle."
    ),
    "windowing": (
        "One fixed-length window per file, matching how the published checkpoint "
        "metrics were produced. No multi-window aggregation is applied, so the 5 s "
        "variant genuinely sees only 5 s and long-file behaviour is not masked."
    ),
    "lofcz": (
        "The lofcz fakeprint detector is pinned to a source revision and a model "
        "revision (see LOFCZ_SOURCE_REVISION and LOFCZ_MODEL_REVISION) plus the "
        "runtime SHA-256 of ai_music_detector.onnx and config.json, all recorded "
        "in the detector config and therefore in the cache key. Audio is loaded "
        "with torchaudio.load, not librosa: resampler differences would otherwise "
        "shift the saturated output probabilities. Only the first 300 s are kept; "
        "short files are scored as-is because the centred STFT pads them, so "
        "there is no minimum-duration failure. The STFT runs on the CPU with "
        "Spectrogram defaults, matching the reference up to ~1e-6 in probability."
    ),
}


def resolve_normalize_mode(audio_config: Mapping) -> str:
    """Map a checkpoint audio config to an AudioDataset normalize mode string.

    See FIDELITY_NOTES["normalize"]: the config field is a boolean while the
    loader argument is a string whose default is "std".
    """
    value = audio_config.get("normalize", True)
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode not in NORMALIZE_MODES:
            raise ValueError(f"unsupported normalize mode in config: {value!r}")
        return mode
    return "std" if bool(value) else "none"


# ---------------------------------------------------------------------------
# 8. Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class FileScore:
    """One detector score for one file, with provenance and timing."""

    sample_id: str
    label: int
    group: str
    generator_family: str
    license_id: str
    transform: str
    score: float
    seconds: float
    cached: bool


@dataclass
class FileFailure:
    """A file the detector could not score."""

    sample_id: str
    error_type: str
    message: str


def runtime_summary(compute_seconds, n_scored: int, total_files=None) -> dict:
    """Per-file latency percentiles plus a full-set runtime extrapolation."""
    measured = np.asarray(list(compute_seconds), dtype=np.float64)
    if measured.size == 0:
        return {
            "n_measured": 0,
            "n_scored": int(n_scored),
            "note": "all scores served from cache; no fresh latency measured",
            "mean_seconds": None,
            "median_seconds": None,
            "p90_seconds": None,
            "max_seconds": None,
            "measured_total_seconds": 0.0,
            "extrapolated_total_seconds": None,
            "extrapolated_total_hours": None,
            "extrapolation_basis_files": int(total_files) if total_files else None,
        }
    mean_seconds = float(measured.mean())
    summary = {
        "n_measured": int(measured.size),
        "n_scored": int(n_scored),
        "note": "",
        "mean_seconds": mean_seconds,
        "median_seconds": float(np.median(measured)),
        "p90_seconds": float(np.percentile(measured, 90)),
        "max_seconds": float(measured.max()),
        "measured_total_seconds": float(measured.sum()),
        "extrapolated_total_seconds": None,
        "extrapolated_total_hours": None,
        "extrapolation_basis_files": int(total_files) if total_files else None,
    }
    if total_files:
        total_seconds = mean_seconds * int(total_files)
        summary["extrapolated_total_seconds"] = float(total_seconds)
        summary["extrapolated_total_hours"] = float(total_seconds / 3600.0)
    return summary


def run_detector(
    detector: Detector,
    rows: Sequence[ManifestRow],
    cache: ScoreCache | None = None,
    progress: Callable[[int, int], None] | None = None,
    total_files_for_extrapolation=None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Score every row with one detector and return grouped metrics.

    Each file is scored independently; no statistic computed over the evaluated
    set is fed back into any score, so results contain no cross-file score
    normalisation and stay valid for a per-file submission path.
    """
    if not rows:
        raise ValueError("no rows to score")
    active_cache = cache or ScoreCache(None, enabled=False)
    detector_key = detector.spec.key
    scores: list[FileScore] = []
    failures: list[FileFailure] = []
    compute_seconds: list[float] = []

    for index, row in enumerate(rows, start=1):
        try:
            if not row.audio_path.is_file():
                raise ScoringFailure(f"audio file not found: {row.audio_path}")
            cache_key = None
            cached_value = None
            if active_cache.enabled:
                cache_key = ScoreCache.make_key(file_digest(row.audio_path), detector_key)
                cached_value = active_cache.get(cache_key)
            if cached_value is not None:
                scores.append(
                    FileScore(
                        sample_id=row.sample_id,
                        label=row.label,
                        group=row.group,
                        generator_family=row.generator_family,
                        license_id=row.license_id,
                        transform=row.transform,
                        score=float(cached_value),
                        seconds=0.0,
                        cached=True,
                    )
                )
            else:
                started = time.perf_counter()
                value = float(detector.score(row.audio_path))
                elapsed = time.perf_counter() - started
                if not np.isfinite(value):
                    raise ScoringFailure(f"detector returned a non-finite score: {value}")
                compute_seconds.append(elapsed)
                if cache_key is not None:
                    active_cache.put(cache_key, value)
                scores.append(
                    FileScore(
                        sample_id=row.sample_id,
                        label=row.label,
                        group=row.group,
                        generator_family=row.generator_family,
                        license_id=row.license_id,
                        transform=row.transform,
                        score=value,
                        seconds=elapsed,
                        cached=False,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a run
            failures.append(
                FileFailure(
                    sample_id=row.sample_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        if progress is not None:
            progress(index, len(rows))

    active_cache.flush()
    if not scores:
        first_error = failures[0].message if failures else "unknown"
        raise ScoringFailure(
            f"detector {detector.spec.name} scored no files; first error: {first_error}"
        )

    labels = np.asarray([item.label for item in scores], dtype=np.int64)
    values = np.asarray([item.score for item in scores], dtype=np.float64)
    groups = [item.group for item in scores]
    families = [item.generator_family for item in scores]

    overall = summarize(labels, values, subset="all")
    return {
        "detector": {
            "name": detector.spec.name,
            "kind": detector.spec.kind,
            "input_kind": detector.spec.input_kind,
            "config": dict(detector.spec.config),
            "cache_key": detector_key,
        },
        "overall": overall.as_dict(),
        "per_group": [item.as_dict() for item in per_group_summaries(labels, values, groups)],
        "leave_one_family_out": [
            item.as_dict() for item in leave_one_generator_family_out(labels, values, families)
        ],
        "bootstrap_eer_ci": grouped_bootstrap_eer_ci(
            labels,
            values,
            groups,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "runtime": runtime_summary(compute_seconds, len(scores), total_files_for_extrapolation),
        "cache": {
            "enabled": active_cache.enabled,
            "hits": active_cache.hits,
            "misses": active_cache.misses,
        },
        "failures": [dict(vars(item)) for item in failures],
        "n_failures": len(failures),
        "scores": [dict(vars(item)) for item in scores],
    }
