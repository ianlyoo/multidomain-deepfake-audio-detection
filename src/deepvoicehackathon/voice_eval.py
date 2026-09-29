"""Offline benchmark harness for the AntiDeepfake speech deepfake detector.

The harness scores one voice detector at a time on a grouped manifest and
reuses the project's exact competition EER implementation, so a reported
number is directly comparable with the music benchmark and with the official
baseline anchor.

Checkpoint provenance
---------------------
checkpoints/wav2vec-large-anti-deepfake, model.safetensors SHA-256
b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5
(nii-yamagishilab/wav2vec-large-anti-deepfake, CC BY-NC-SA 4.0), paper
"Post-training for Deepfake Speech Detection" (arXiv:2506.21090).

The bundled config.json is a stale Hugging Face Wav2Vec2ForCTC config and must
not be used to build the model: the tensor names in model.safetensors are
'm_ssl.model.*' plus 'proj_fc.weight'/'proj_fc.bias', i.e. a fairseq
Wav2Vec2Model front-end with a 2-way linear head. SSL_CONFIG_KWARGS below is
transcribed verbatim from the model card's inference snippet, which is the only
published description of the architecture that matches those tensors.

Preprocessing is transcribed from the same snippet:

* torchaudio.load then wav.mean(dim=0) -> mono by channel average
* torchaudio.functional.resample(wav, sr, 16000) -> 16 kHz
* torch.nn.functional.layer_norm(wav, wav.shape) -> zero-mean/unit-variance
  over the whole waveform (eps 1e-5), applied after resampling
* forward: features_only (mask=False) -> [B, T, 1024] -> transpose -> adaptive
  average pool over time -> Linear(1024, 2)
* score: softmax(logits)[0] is the fake probability, [1] is real

Score direction is therefore FAKE_CLASS_INDEX = 0, and this module always
returns "higher means more likely fake", matching music_eval and the
FILE_FAKE_PROB / VOICE_FAKE_PROB convention of the competition.

Runtime boundary
----------------
The front-end is rebuilt by the torch-native loader in _antideepfake_frontend,
transcribed operation-for-operation from fairseq 0.12.2 (which does not install
on this Python 3.11 / torch 2.7.1 environment; the card pins Python 3.9 and
torch 1.x). Every heavy import is therefore lazy and every failure is reported
as DetectorUnavailableError with the exact remediation, so metrics, cache,
fusion and CLI wiring stay testable and reviewable even where the weights
cannot be executed. See environment_report().
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._antideepfake_frontend import (
    NATIVE_FRONTEND_VERSION,
    PRETRAINING_ONLY_KEYS,
    audit_checkpoint_keys,
    build_native_detector_class,
    format_audit_error,
    read_safetensors_header,
)

# The runner, cache, manifest loader and metric helpers are shared with the
# music benchmark on purpose: one EER/bootstrap implementation, one cache
# format, one report shape. music_eval owns them; this module only adds the
# voice detector, its preprocessing and score fusion.
from .music_eval import (  # noqa: F401 - several names are re-exported for the CLI
    Detector,
    DetectorUnavailableError,
    ManifestColumns,
    ManifestRow,
    ScoreCache,
    ScoringFailure,
    config_fingerprint,
    file_digest,
    grouped_bootstrap_eer_ci,
    leave_one_generator_family_out,
    load_manifest,
    per_group_summaries,
    run_detector,
    runtime_summary,
    sample_rows,
    summarize,
)

ANTIDEEPFAKE_MODEL_ID = "nii-yamagishilab/wav2vec-large-anti-deepfake"
ANTIDEEPFAKE_REPOSITORY = "https://github.com/nii-yamagishilab/AntiDeepfake"
ANTIDEEPFAKE_PAPER = "https://arxiv.org/abs/2506.21090"
ANTIDEEPFAKE_LICENSE = "CC BY-NC-SA 4.0 (research/educational, non-commercial)"
CHECKPOINT_SHA256 = "b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5"
CHECKPOINT_WEIGHT_FILE = "model.safetensors"

VOICE_SAMPLE_RATE = 16_000
LAYER_NORM_EPS = 1e-5
# wav2vec2 conv front-end: kernels 10,3,3,3,3,2,2 with strides 5,2,2,2,2,2,2.
# Fewer than 400 samples (25 ms) cannot produce a single output frame.
MIN_INPUT_SAMPLES = 400

# Memory-safe hard ceiling for a single forward pass. The card is exact for
# arbitrary lengths, so the default stays uncapped (None); any effective model
# input larger than this raises ScoringFailure instead of risking an OOM in
# the 24-layer transformer. The submission path caps to 30 s (480k samples).
SUBMISSION_MAX_AUDIO_SECONDS = 30.0
MAX_SAFE_INPUT_SECONDS = 60.0
MAX_SAFE_INPUT_SAMPLES = VOICE_SAMPLE_RATE * int(MAX_SAFE_INPUT_SECONDS)

FAKE_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1
NUM_CLASSES = 2
SSL_OUTPUT_DIM = 1024

CROP_MODES = ("none", "head", "center", "tail")
INPUT_KINDS = ("mixture", "voice_stem")
RESAMPLERS = ("torchaudio", "librosa")
FUSION_METHODS = ("mean", "max", "min", "rank_mean")

FAIRSEQ_REQUIREMENT = "fairseq==0.12.2"
FAIRSEQ_HINT = (
    "The AntiDeepfake front-end is a fairseq Wav2Vec2Model. Install "
    + FAIRSEQ_REQUIREMENT
    + " in a Python 3.9 environment with torch<2 (per the model card), or port "
    "the weights to a fairseq-free front-end. fairseq 0.12.2 does not build on "
    "Python 3.11."
)

# Verbatim from the model card's SSLModel.__init__ Wav2Vec2Config(...) call.
SSL_CONFIG_KWARGS: dict = {
    "quantize_targets": True,
    "extractor_mode": "layer_norm",
    "layer_norm_first": True,
    "final_dim": 768,
    "latent_temp": (2.0, 0.1, 0.999995),
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "conv_bias": True,
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "feature_grad_mult": 1.0,
}

VOICE_FIDELITY_NOTES: dict = {
    "config_json": (
        "The checkpoint's config.json declares Wav2Vec2ForCTC and is unused: it "
        "does not describe the m_ssl.model.* / proj_fc tensors actually stored. "
        "SSL_CONFIG_KWARGS from the model card is authoritative."
    ),
    "normalisation": (
        "layer_norm(wav, wav.shape) is computed over the exact waveform handed "
        "to the model, so a duration cap changes the normalisation statistics. "
        "max_audio_seconds=None (card-exact, arbitrary length) is the default."
    ),
    "resampler": (
        "The card uses torchaudio.functional.resample. librosa is accepted as a "
        "fallback and recorded in the detector config, so cached scores from the "
        "two resamplers never collide."
    ),
    "pooling": (
        "AdaptiveAvgPool1d(1) over all frames: one score per file, no windowing "
        "and no cross-file statistics, so results stay valid for a per-file "
        "submission path."
    ),
    "score_direction": (
        "softmax(logits)[0] is fake and [1] is real, per the card's printout. "
        "This module returns the fake probability so higher always means faker."
    ),
    "native_frontend": (
        "The torch-native loader in _antideepfake_frontend transcribes fairseq "
        "0.12.2 operation-for-operation (pre-norm layers, manual pos-conv "
        "weight norm, SamePad trim, pad-to-multiple-of-2) and loads the "
        "checkpoint with strict=True after dropping exactly the 8 audited "
        "pretraining-only tensors; any other key or shape mismatch is an error."
    ),
    "weight_norm": (
        "The native front-end declares encoder.pos_conv.0.weight_g/weight_v/"
        "bias verbatim, matching the checkpoint, so strict loading applies no "
        "remapping (remapped_weight_norm_keys == []). remap_weight_norm_keys "
        "remains only as a legacy helper for torch>=2.1 parametrized modules "
        "and is not used by this loader."
    ),
}


# ---------------------------------------------------------------------------
# 1. Preprocessing (card-exact, numpy so it is testable without torch)
# ---------------------------------------------------------------------------

def to_mono(audio) -> np.ndarray:
    """Average channels like the card's wav.mean(dim=0).

    Accepts (T,) or (channels, T); 3-D or higher input is rejected. A
    (T, channels) layout is not detected because a 2-sample mono clip and a
    2-channel frame are indistinguishable, so callers must pass channel-first
    data exactly as torchaudio.load returns it.
    """
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        mono = array.mean(axis=0, dtype=np.float32)
    else:
        raise ValueError("audio must be 1-D (T,) or 2-D (channels, T)")
    mono = np.ascontiguousarray(mono.reshape(-1), dtype=np.float32)
    if mono.size == 0:
        raise ValueError("audio is empty")
    return mono


def crop_waveform(audio, max_samples=None, crop_mode: str = "head") -> np.ndarray:
    """Deterministically limit waveform length; never pads.

    Padding is deliberately absent: the card feeds arbitrary-length audio and
    zero padding would shift the layer_norm statistics of short clips.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio is empty")
    if max_samples is None or crop_mode == "none":
        return waveform.copy()
    limit = int(max_samples)
    if limit <= 0:
        raise ValueError("max_samples must be positive")
    if waveform.size <= limit:
        return waveform.copy()
    if crop_mode == "head":
        start = 0
    elif crop_mode == "center":
        start = (waveform.size - limit) // 2
    else:
        start = waveform.size - limit
    return waveform[start : start + limit].copy()


def layer_norm_waveform(audio, eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """Reproduce torch.nn.functional.layer_norm(wav, wav.shape).

    Biased variance over the whole waveform, no affine term. Accumulating in
    float64 keeps long files stable; the result is cast back to float32.
    """
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio is empty")
    values = waveform.astype(np.float64, copy=False)
    mean = values.mean()
    variance = values.var()
    normalised = (values - mean) / np.sqrt(variance + float(eps))
    return np.ascontiguousarray(normalised, dtype=np.float32)


def prepare_voice_waveform(
    audio,
    max_samples=None,
    crop_mode: str = "head",
    apply_layer_norm: bool = True,
    eps: float = LAYER_NORM_EPS,
) -> np.ndarray:
    """Mono, optional crop, then layer_norm, in the card's order.

    The crop happens before normalisation so the statistics describe exactly
    the samples the model sees.
    """
    mono = to_mono(audio)
    cropped = crop_waveform(mono, max_samples=max_samples, crop_mode=crop_mode)
    if not apply_layer_norm:
        return cropped
    return layer_norm_waveform(cropped, eps=eps)


# ---------------------------------------------------------------------------
# 2. Score direction
# ---------------------------------------------------------------------------

def softmax_last_axis(logits) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    array = np.asarray(logits, dtype=np.float64)
    if array.size == 0:
        raise ValueError("logits are empty")
    if not np.isfinite(array).all():
        raise ValueError("logits contain NaN or infinity")
    shifted = array - array.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def fake_probability_from_logits(logits) -> float:
    """Fake probability for one item: softmax(logits)[FAKE_CLASS_INDEX]."""
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("expected logits of shape (2,) or (1, 2)")
    if array.shape[-1] != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} classes, got {array.shape[-1]}")
    probability = float(softmax_last_axis(array)[0, FAKE_CLASS_INDEX])
    if not np.isfinite(probability):
        raise ScoringFailure("non-finite fake probability")
    return probability


# ---------------------------------------------------------------------------
# 3. Detector identity and cache key
# ---------------------------------------------------------------------------

@dataclass
class VoiceDetectorSpec:
    """Identity and configuration of one scored voice detector variant.

    Mirrors music_eval.DetectorSpec (same key shape, so the on-disk cache
    format is shared) but allows voice-specific input kinds.
    """

    name: str
    kind: str
    input_kind: str = "mixture"
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_kind not in INPUT_KINDS:
            raise ValueError(f"input_kind must be one of {INPUT_KINDS}")

    @property
    def key(self) -> str:
        payload = dict(self.config)
        payload["__kind__"] = self.kind
        payload["__input__"] = self.input_kind
        return f"{self.name}|{self.input_kind}|{config_fingerprint(payload)}"


# ---------------------------------------------------------------------------
# 4. Lazy dependency and checkpoint handling
# ---------------------------------------------------------------------------

def _require_module(module_name: str, hint: str):
    """Import a heavy optional dependency, or explain precisely why not.

    fairseq can fail with more than ImportError on modern Python (its
    dataclass/omegaconf setup raises at import time), so every exception is
    converted into DetectorUnavailableError.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - see docstring
        raise DetectorUnavailableError(
            f"{module_name} is required for this detector but is not importable "
            f"({type(exc).__name__}: {exc}). {hint}"
        ) from exc


DEPENDENCY_MODULES = ("torch", "torchaudio", "fairseq", "safetensors", "soundfile", "librosa")


def module_availability(names: Sequence[str] = DEPENDENCY_MODULES) -> dict:
    """Report importability of every dependency the detector may need."""
    import importlib.util

    report: dict = {}
    for name in names:
        try:
            report[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            report[name] = False
    return report


def environment_report() -> dict:
    """Environment diagnostics used when a voice detector cannot run."""
    availability = module_availability()
    blockers: list[str] = []
    # fairseq is intentionally not required: the torch-native frontend in
    # _antideepfake_frontend replaces it. Presence is reported, never a blocker.
    if not availability.get("torch", False):
        blockers.append("torch is not importable; no checkpoint can be executed.")
    if not availability.get("safetensors", False):
        blockers.append("safetensors is not importable; the weight file cannot be read.")
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "modules": availability,
        "blockers": blockers,
        "checkpoint_sha256_expected": CHECKPOINT_SHA256,
        "model_id": ANTIDEEPFAKE_MODEL_ID,
        "license": ANTIDEEPFAKE_LICENSE,
    }


def verify_checkpoint_digest(path, expected: str = CHECKPOINT_SHA256, chunk_bytes: int = 8 * 1024 * 1024) -> dict:
    """Hash a weight file and compare it with the pinned digest."""
    target = Path(path)
    if not target.is_file():
        raise DetectorUnavailableError(f"checkpoint weights not found: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    actual = digest.hexdigest()
    return {
        "path": str(target),
        "expected": str(expected),
        "actual": actual,
        "matches": actual == str(expected),
        "size_bytes": target.stat().st_size,
    }


WEIGHT_NORM_ALIASES = {
    "weight_g": "parametrizations.weight.original0",
    "weight_v": "parametrizations.weight.original1",
}


def remap_weight_norm_keys(state_dict: Mapping[str, object], target_keys: Sequence[str]):
    """Bridge legacy weight_g/weight_v and torch>=2.1 parametrisations.

    Returns the remapped state dict and the applied (old, new) pairs. Keys the
    target model already expects are left untouched, so this is a no-op when
    the checkpoint and the installed torch agree.
    """
    expected = {str(key) for key in target_keys}
    reverse = {new: old for old, new in WEIGHT_NORM_ALIASES.items()}
    remapped: dict = {}
    applied: list = []
    for key, value in state_dict.items():
        name = str(key)
        if name not in expected:
            base, _, suffix = name.rpartition(".")
            candidate = None
            if suffix in WEIGHT_NORM_ALIASES and base:
                candidate = f"{base}.{WEIGHT_NORM_ALIASES[suffix]}"
            else:
                for parametrised, legacy in reverse.items():
                    if name.endswith("." + parametrised):
                        candidate = name[: -len(parametrised)] + legacy
                        break
            if candidate is not None and candidate in expected:
                applied.append((name, candidate))
                name = candidate
        remapped[name] = value
    return remapped, applied


def build_ssl_config(fairseq_wav2vec):
    """Build the card's Wav2Vec2Config from SSL_CONFIG_KWARGS."""
    return fairseq_wav2vec.Wav2Vec2Config(**SSL_CONFIG_KWARGS)


def build_deepfake_detector_class(torch, fairseq_wav2vec):
    """Define the card's SSL front-end plus FC head at call time.

    The classes are built lazily so importing this module never needs torch or
    fairseq, which is what keeps the harness unit-testable.
    """

    class SSLModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = fairseq_wav2vec.Wav2Vec2Model(build_ssl_config(fairseq_wav2vec))

        def extract_feat(self, input_data):
            if input_data.ndim == 3:
                input_data = input_data[:, :, 0]
            return self.model(input_data, mask=False, features_only=True)["x"]

    class DeepfakeDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ssl_orig_output_dim = SSL_OUTPUT_DIM
            self.num_classes = NUM_CLASSES
            self.m_ssl = SSLModel()
            self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
            self.proj_fc = torch.nn.Linear(
                in_features=self.ssl_orig_output_dim, out_features=self.num_classes
            )

        def forward(self, wav):
            emb = self.m_ssl.extract_feat(wav)
            emb = emb.transpose(1, 2)
            pooled = self.adap_pool1d(emb).squeeze(-1)
            return self.proj_fc(pooled)

    return DeepfakeDetector


# ---------------------------------------------------------------------------
# 5. Audio loading
# ---------------------------------------------------------------------------

def resolve_resampler(preference: str = "auto") -> str:
    """Pick the resampler backend, preferring the card's torchaudio."""
    if preference not in ("auto",) + RESAMPLERS:
        raise ValueError(f"resampler must be 'auto' or one of {RESAMPLERS}")
    availability = module_availability(("torchaudio", "librosa"))
    if preference != "auto":
        if not availability.get(preference, False):
            raise DetectorUnavailableError(
                f"resampler {preference!r} was requested but the module is not importable."
            )
        return preference
    if availability.get("torchaudio", False):
        return "torchaudio"
    if availability.get("librosa", False):
        return "librosa"
    raise DetectorUnavailableError(
        "neither torchaudio nor librosa is importable; audio cannot be decoded. "
        "Install the inference extra: uv sync --extra inference"
    )


def load_voice_waveform(
    audio_path,
    target_sample_rate: int = VOICE_SAMPLE_RATE,
    resampler: str = "auto",
) -> np.ndarray:
    """Load mono float32 audio at 16 kHz following the card's pipeline."""
    backend = resolve_resampler(resampler)
    if backend == "torchaudio":
        torch = _require_module("torch", "Install torch to decode audio.")
        torchaudio = _require_module(
            "torchaudio", "Install the inference extra: uv sync --extra inference"
        )
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:  # noqa: BLE001 - per-file decode failure
            raise ScoringFailure(f"failed to decode {audio_path}: {exc}") from exc
        with torch.no_grad():
            mono = waveform.mean(dim=0)
            if int(sample_rate) != int(target_sample_rate):
                mono = torchaudio.functional.resample(
                    mono, int(sample_rate), int(target_sample_rate)
                )
        audio = mono.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        librosa = _require_module(
            "librosa", "Install the inference extra: uv sync --extra inference"
        )
        try:
            audio, _ = librosa.load(
                str(audio_path), sr=int(target_sample_rate), mono=True, dtype=np.float32
            )
        except Exception as exc:  # noqa: BLE001 - per-file decode failure
            raise ScoringFailure(f"failed to decode {audio_path}: {exc}") from exc
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ScoringFailure(f"invalid audio: {audio_path}")
    return audio


class DemucsVoiceStemProvider:
    """HTDemucs vocals stem at 16 kHz, mirroring the official baseline.

    Verbatim vocals branch of separate_voice_and_music from
    baseline/official/script.py: demucs.separate.load_track decoding, mono
    mean/std normalisation, digital-silence shortcut (std < 1e-8), apply_model
    with shifts=0/split=True/overlap=0.25, sources*std+mean rescale, vocals
    channel mean, then resample to 16 kHz. self.sample_rate stands in for the
    baseline AUDIO_SAMPLE_RATE (both are 16_000 by default). A fused voice
    score is therefore computed on the same signal the official submission uses.
    """

    def __init__(self, repo_dir=None, device: str = "cpu", sample_rate: int = VOICE_SAMPLE_RATE) -> None:
        self._torch = _require_module("torch", "Install torch to run HTDemucs.")
        self._torchaudio = _require_module(
            "torchaudio", "Install the inference extra: uv sync --extra inference"
        )
        demucs_pretrained = _require_module(
            "demucs.pretrained", "Install the inference extra: uv sync --extra inference"
        )
        self._apply = _require_module(
            "demucs.apply", "Install the inference extra: uv sync --extra inference"
        )
        self._separate = _require_module(
            "demucs.separate", "Install the inference extra: uv sync --extra inference"
        )
        self.sample_rate = int(sample_rate)
        self.device = self._torch.device(device)
        repo = Path(repo_dir).resolve() if repo_dir is not None else None
        if repo is not None and not repo.is_dir():
            raise DetectorUnavailableError(f"HTDemucs repo directory not found: {repo}")
        # PyTorch 2.6 changed torch.load's default to weights_only=True. The
        # pinned, locally verified Demucs checkpoint contains its model class,
        # so mirror the competition baseline's trusted-checkpoint loader.
        original_load = self._torch.load

        def load_trusted_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        self._torch.load = load_trusted_checkpoint
        try:
            self.model = demucs_pretrained.get_model("htdemucs", repo=repo)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(f"failed to load HTDemucs: {exc}") from exc
        finally:
            self._torch.load = original_load
        self.model.to(self.device).eval()
        if "vocals" not in self.model.sources:
            raise DetectorUnavailableError(
                f"HTDemucs model exposes no vocals source: {self.model.sources}"
            )

    def __call__(self, audio_path) -> np.ndarray:
        torch = self._torch
        model = self.model
        try:
            waveform = self._separate.load_track(
                str(audio_path), model.audio_channels, model.samplerate
            ).float()
        except Exception as exc:  # noqa: BLE001 - per-file decode failure
            raise ScoringFailure(f"failed to decode {audio_path}: {exc}") from exc
        mono_waveform = waveform.mean(0)
        mean = mono_waveform.mean()
        std = mono_waveform.std()
        if float(std) < 1e-8:
            length = round(waveform.shape[-1] * self.sample_rate / model.samplerate)
            return np.zeros(max(1, length), dtype=np.float32)
        normalized_waveform = (waveform - mean) / std
        with torch.inference_mode():
            sources = self._apply.apply_model(
                model,
                normalized_waveform[None],
                device=self.device,
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        sources = sources * std + mean
        vocal_index = model.sources.index("vocals")
        voice_audio = sources[vocal_index].mean(0, keepdim=True)
        voice_audio = self._torchaudio.functional.resample(
            voice_audio, model.samplerate, self.sample_rate
        )[0]
        stem = voice_audio.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if stem.size == 0 or not np.isfinite(stem).all():
            raise ScoringFailure(f"invalid vocals stem: {audio_path}")
        return stem


# ---------------------------------------------------------------------------
# 6. AntiDeepfake detector
# ---------------------------------------------------------------------------

def check_voice_input_limits(n_samples, max_audio_samples=None) -> None:
    """Preflight guard: deterministic ScoringFailure instead of OOM or cryptic matmul errors."""
    configured = int(max_audio_samples) if max_audio_samples is not None else None
    if configured is not None and configured < MIN_INPUT_SAMPLES:
        raise ScoringFailure(
            f"configured input cap ({configured} samples) is below the "
            f"{MIN_INPUT_SAMPLES}-sample minimum for one wav2vec2 frame."
        )
    effective = min(int(n_samples), configured) if configured is not None else int(n_samples)
    if effective < MIN_INPUT_SAMPLES:
        raise ScoringFailure(
            f"audio shorter than {MIN_INPUT_SAMPLES} samples ({int(n_samples)}); "
            "wav2vec2 cannot produce a frame."
        )
    if effective > MAX_SAFE_INPUT_SAMPLES:
        raise ScoringFailure(
            f"effective input ({effective} samples) exceeds the memory-safe limit "
            f"of {MAX_SAFE_INPUT_SAMPLES} samples ({MAX_SAFE_INPUT_SECONDS:.0f} s at 16 kHz); "
            f"configure max_audio_seconds<={MAX_SAFE_INPUT_SECONDS:.0f} (submission uses "
            f"{SUBMISSION_MAX_AUDIO_SECONDS:.0f} s)."
        )


def drop_pretraining_keys(state_dict):
    """Remove exactly the audited pretraining-only tensors; anything missing is an error."""
    for pretraining_key in PRETRAINING_ONLY_KEYS:
        if pretraining_key not in state_dict:
            raise DetectorUnavailableError(
                f"checkpoint is missing the audited pretraining tensor {pretraining_key!r}; "
                "refusing to silently continue with a partial weight set."
            )
        state_dict.pop(pretraining_key)
    return state_dict


class AntiDeepfakeVoiceDetector(Detector):
    """AntiDeepfake Wav2Vec2-Large scored exactly as the model card specifies.

    One forward pass per file over the full (optionally capped) waveform, no
    windowing, no cross-file statistics. Returns the fake probability.

    Every heavy dependency is imported inside __init__, so construction is the
    only operation that can raise DetectorUnavailableError; the surrounding
    harness stays importable and testable without fairseq.
    """

    def __init__(
        self,
        checkpoint_dir,
        name: str = "antideepfake-wav2vec-large",
        device: str = "cpu",
        max_audio_seconds: float | None = None,
        crop_mode: str = "head",
        input_kind: str = "mixture",
        stem_provider=None,
        resampler: str = "auto",
        verify_digest: bool = False,
        expected_sha256: str = CHECKPOINT_SHA256,
    ) -> None:
        checkpoint_path = Path(checkpoint_dir)
        weight_path = checkpoint_path / CHECKPOINT_WEIGHT_FILE
        if not weight_path.is_file():
            raise DetectorUnavailableError(
                f"AntiDeepfake weights not found: {weight_path}. Download "
                f"{ANTIDEEPFAKE_MODEL_ID} into {checkpoint_path} first."
            )
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")
        if input_kind not in INPUT_KINDS:
            raise ValueError(f"input_kind must be one of {INPUT_KINDS}")
        if input_kind == "voice_stem" and stem_provider is None:
            raise ValueError("voice_stem input requires a stem provider")
        if max_audio_seconds is not None and float(max_audio_seconds) <= 0:
            raise ValueError("max_audio_seconds must be positive")

        max_audio_samples = (
            int(round(float(max_audio_seconds) * VOICE_SAMPLE_RATE))
            if max_audio_seconds is not None
            else None
        )
        if max_audio_samples is not None and max_audio_samples > MAX_SAFE_INPUT_SAMPLES:
            raise ValueError(
                f"max_audio_seconds ({float(max_audio_seconds)} s) exceeds the memory-safe "
                f"limit of {MAX_SAFE_INPUT_SECONDS:.0f} s ({MAX_SAFE_INPUT_SAMPLES} samples)."
            )

        digest_check = None
        if verify_digest:
            digest_check = verify_checkpoint_digest(weight_path, expected_sha256)
            if not digest_check["matches"]:
                raise DetectorUnavailableError(
                    f"checkpoint digest mismatch for {weight_path}: expected "
                    f"{digest_check['expected']}, found {digest_check['actual']}"
                )

        # Checkpoint identity (pure-Python header audit) comes before resampler
        # resolution so a corrupt/foreign weight file is reported deterministically
        # even where no audio backend is installed.
        try:
            header_shapes = read_safetensors_header(weight_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(
                f"failed to read the safetensors header of {weight_path}: {exc}"
            ) from exc
        coverage = audit_checkpoint_keys(header_shapes)
        if not coverage["complete"]:
            raise DetectorUnavailableError(format_audit_error(coverage, weight_path))

        self.resampler = resolve_resampler(resampler)
        self._torch = _require_module("torch", "Install torch to run AntiDeepfake.")
        safetensors_torch = _require_module(
            "safetensors.torch", "Install safetensors to read model.safetensors."
        )

        torch = self._torch
        detector_class = build_native_detector_class(torch)
        try:
            model = detector_class()
        except Exception as exc:  # noqa: BLE001 - torch version conflict
            raise DetectorUnavailableError(
                "failed to build the AntiDeepfake architecture "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        try:
            state_dict = safetensors_torch.load_file(str(weight_path), device="cpu")
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear load error
            raise DetectorUnavailableError(f"failed to read {weight_path}: {exc}") from exc
        drop_pretraining_keys(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise DetectorUnavailableError(
                f"AntiDeepfake weights do not match the rebuilt architecture: {exc}. "
                "See VOICE_FIDELITY_NOTES."
            ) from exc
        head = model.proj_fc
        if tuple(head.weight.shape) != (NUM_CLASSES, SSL_OUTPUT_DIM):
            raise DetectorUnavailableError(
                f"unexpected classifier shape {tuple(head.weight.shape)}; "
                f"expected {(NUM_CLASSES, SSL_OUTPUT_DIM)}"
            )

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        self.stem_provider = stem_provider
        self.crop_mode = crop_mode
        self.sample_rate = VOICE_SAMPLE_RATE
        self.max_audio_seconds = float(max_audio_seconds) if max_audio_seconds is not None else None
        self.max_audio_samples = max_audio_samples
        self.remapped_weight_norm_keys = []
        self.key_coverage = coverage
        self.digest_check = digest_check
        self.spec = VoiceDetectorSpec(
            name=name,
            kind="antideepfake_wav2vec",
            input_kind=input_kind,
            config={
                "model_id": ANTIDEEPFAKE_MODEL_ID,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": str(expected_sha256),
                "frontend": "native-torch",
                "frontend_version": NATIVE_FRONTEND_VERSION,
                "sample_rate": self.sample_rate,
                "layer_norm_eps": LAYER_NORM_EPS,
                "max_audio_seconds": self.max_audio_seconds,
                "crop_mode": crop_mode if self.max_audio_samples is not None else None,
                "resampler": self.resampler,
                "pooling": "adaptive_avg_pool_1d",
                "fake_class_index": FAKE_CLASS_INDEX,
                "separation": "htdemucs-vocals" if input_kind == "voice_stem" else "none",
            },
        )

    def input_waveform(self, audio_path) -> np.ndarray:
        """Decoded 16 kHz mono signal, before crop and normalisation."""
        if self.spec.input_kind == "voice_stem":
            stem = np.asarray(self.stem_provider(audio_path), dtype=np.float32).reshape(-1)
            if stem.size == 0:
                raise ScoringFailure(f"empty vocals stem: {audio_path}")
            return stem
        return load_voice_waveform(audio_path, self.sample_rate, self.resampler)

    def prepare(self, audio) -> np.ndarray:
        """Card-exact model input for one file."""
        check_voice_input_limits(np.asarray(audio).size, self.max_audio_samples)
        window = prepare_voice_waveform(
            audio, max_samples=self.max_audio_samples, crop_mode=self.crop_mode
        )
        check_voice_input_limits(window.size, self.max_audio_samples)
        return window

    def score(self, audio_path) -> float:
        audio = self.input_waveform(audio_path)
        try:
            check_voice_input_limits(audio.size, self.max_audio_samples)
        except ScoringFailure as exc:
            raise ScoringFailure(f"{exc}: {audio_path}") from exc
        try:
            window = self.prepare(audio)
        except ScoringFailure as exc:
            raise ScoringFailure(f"{exc}: {audio_path}") from exc
        torch = self._torch
        tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        return fake_probability_from_logits(logits.float().cpu().numpy())

    def close(self) -> None:
        self.model = None


# ---------------------------------------------------------------------------
# 7. Fusion against the official DF-Arena voice score
# ---------------------------------------------------------------------------

def _score_matrix(components: Mapping[str, Sequence[float]]):
    if not components:
        raise ValueError("no score components to fuse")
    names = list(components)
    vectors = [np.asarray(components[name], dtype=np.float64).reshape(-1) for name in names]
    lengths = {int(vector.size) for vector in vectors}
    if len(lengths) != 1:
        raise ValueError("all score components must have the same length")
    if lengths == {0}:
        raise ValueError("score components must be non-empty")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("score components contain NaN or infinity")
    return names, matrix


def _resolve_weights(names: Sequence[str], weights: Mapping[str, float] | None) -> np.ndarray:
    if weights is None:
        return np.full(len(names), 1.0 / len(names), dtype=np.float64)
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"missing fusion weights for {missing}")
    array = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if (array < 0).any():
        raise ValueError("fusion weights must be non-negative")
    total = array.sum()
    if total <= 0:
        raise ValueError("fusion weights must sum to a positive value")
    return array / total


def rank_normalise(scores) -> np.ndarray:
    """Average-rank transform scaled to (0, 1]; ties share a rank."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores are empty")
    from scipy.stats import rankdata

    return np.asarray(rankdata(values, method="average"), dtype=np.float64) / values.size


def fuse_scores(
    components: Mapping[str, Sequence[float]],
    method: str = "mean",
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Combine per-file detector scores into one fused score vector.

    mean/max/min are per-file and stay valid for a submission. rank_mean uses
    the evaluated set's ordering, so it is analysis-only; see
    fusion_is_submission_safe.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}")
    names, matrix = _score_matrix(components)
    if method in ("mean", "rank_mean"):
        weight_vector = _resolve_weights(names, weights)
        working = matrix
        if method == "rank_mean":
            working = np.vstack([rank_normalise(row) for row in matrix])
        fused = (weight_vector[:, None] * working).sum(axis=0)
    elif method == "max":
        fused = matrix.max(axis=0)
    else:
        fused = matrix.min(axis=0)
    return np.clip(fused, 0.0, 1.0)


def fusion_is_submission_safe(method: str) -> bool:
    """True when a fusion method needs no statistics across evaluated files."""
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}")
    return method != "rank_mean"


def fusion_report(
    components: Mapping[str, Sequence[float]],
    labels,
    groups,
    families=None,
    methods: Sequence[str] = ("mean", "max"),
    weights: Mapping[str, float] | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Grouped metrics for each component and for each fusion method."""
    names, _ = _score_matrix(components)
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    group_keys = [str(group) for group in groups]
    family_keys = [
        str(family) for family in (families if families is not None else ["unknown"] * y_true.size)
    ]

    def block(label: str, values, submission_safe: bool) -> dict:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return {
            "name": label,
            "submission_safe": bool(submission_safe),
            "overall": summarize(y_true, array, subset="all").as_dict(),
            "bootstrap_eer_ci": grouped_bootstrap_eer_ci(
                y_true, array, group_keys, n_resamples=bootstrap_resamples, seed=bootstrap_seed
            ),
            "leave_one_family_out": [
                item.as_dict()
                for item in leave_one_generator_family_out(y_true, array, family_keys)
            ],
        }

    entries = [block(name, components[name], True) for name in names]
    fused = []
    for method in methods:
        values = fuse_scores(components, method=method, weights=weights)
        entry = block(f"fusion:{method}", values, fusion_is_submission_safe(method))
        entry["method"] = method
        entry["components"] = list(names)
        entry["weights"] = (
            {name: float(value) for name, value in zip(names, _resolve_weights(names, weights))}
            if method in ("mean", "rank_mean")
            else None
        )
        fused.append(entry)
    return {"components": entries, "fusions": fused, "n_files": int(y_true.size)}
