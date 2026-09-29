"""Offline, hash-bound MERT-v1-95M, one center five-second embedding per file.

This module can be embedded verbatim: no project imports, Hub APIs, downloads,
disk writes, model cache, or global device/thread configuration. The caller owns
model lifetime and must finish/unload the MERT prepass before loading old models.
Author assets: m-a-p/MERT-v1-95M, CC-BY-NC-4.0; retain its README attribution.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

REPO_ID = "m-a-p/MERT-v1-95M"
REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
LICENSE = "CC-BY-NC-4.0"
SAMPLE_RATE = 24000
CROP_SAMPLES = 120000
EMBEDDING_SIZE = 768
TRANSFORMERS_VERSION = "4.57.6"
STATE_TENSORS = 211
STATE_VALUES = 94371712
FILE_SHA256 = {
    "config.json": "ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32",
    "configuration_MERT.py": "ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03",
    "modeling_MERT.py": "6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c",
    "preprocessor_config.json": "cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f",
    "pytorch_model.bin": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
    "README.md": "4bd95e900185d8b393785c0038aa662355a904e5a98ec692003b8011c802455d",
}
DERIVED_MODEL_SHA256 = "7fb1349737b12194a28a41ce960ecd9098dad9422f0c24cab94c4528fcdd57e6"
OPTIONAL_CQT_WARNING = "WARNING: feature_extractor_cqt requires the libray 'nnAudio'"
WEIGHT_KEY_MAP = {
    "encoder.pos_conv_embed.conv.weight_g":
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v":
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


class MERTError(RuntimeError):
    """Invalid assets, checkpoint, runtime, processor, or model output; no fallback."""


def _verified_sources(root: Path) -> dict[str, bytes]:
    """Read and bind source bytes once; weights are bound on their open handle."""
    sources = {}
    for name, expected in FILE_SHA256.items():
        if name == "pytorch_model.bin":
            continue
        path = root / name
        if path.stat().st_size > 1_000_000:
            raise MERTError(f"oversized MERT source: {name}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise MERTError(f"MERT SHA256 mismatch: {name}")
        sources[name] = payload
    return sources


def _checkpoint(root: Path):
    import torch

    with (root / "pytorch_model.bin").open("rb") as handle:
        expected = FILE_SHA256["pytorch_model.bin"]
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise MERTError("MERT SHA256 mismatch: pytorch_model.bin")
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
        handle.seek(0)
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise MERTError("MERT weights changed during load")
    if type(state) is not dict or len(state) != STATE_TENSORS:
        raise MERTError("expected exact211 raw checkpoint tensors")
    if any(type(k) is not str or type(v) is not torch.Tensor
           or v.layout != torch.strided or v.device.type != "cpu"
           or v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
           for k, v in state.items()):
        raise MERTError("checkpoint must contain finite dense CPU float32 tensors")
    if sum(v.numel() for v in state.values()) != STATE_VALUES:
        raise MERTError("checkpoint value count mismatch")
    return state


def _exec_author_model(derived, namespace, *, cqt_disabled):
    """Suppress only the author's known unused-CQT print; preserve all else.

    Capture is scoped to this local import. Python warnings/stderr are untouched;
    unknown stdout is replayed even when author execution raises an exception.
    """
    import io
    import sys

    captured = io.StringIO()
    suppressed = 0
    try:
        with redirect_stdout(captured):
            exec(compile(derived, "modeling_MERT.py", "exec"), namespace)
    finally:
        for line in captured.getvalue().splitlines(keepends=True):
            if cqt_disabled and line.rstrip("\r\n") == OPTIONAL_CQT_WARNING:
                suppressed += 1
            else:
                sys.stdout.write(line)
    return suppressed


def _author_classes(sources):
    """Only replace the single relative import; no sys.path/sys.modules changes.

    Both original files and the derived execution bytes have independent pins.
    All author model/config bodies remain byte-identical. Isolated dictionaries
    retain class globals without importing neighboring files or remote modules.
    """
    config_namespace = {"__name__": "_mert95_pinned_configuration"}
    exec(compile(sources["configuration_MERT.py"], "configuration_MERT.py", "exec"),
         config_namespace)
    original = sources["modeling_MERT.py"]
    target = b"from .configuration_MERT import MERTConfig"
    if original.count(target) != 1:
        raise MERTError("expected exactly one local MERT configuration import")
    derived = original.replace(target, b"# MERTConfig injected from authenticated local configuration.")
    if hashlib.sha256(derived).hexdigest() != DERIVED_MODEL_SHA256:
        raise MERTError("derived MERT code SHA256 mismatch")
    model_namespace = {"__name__": "_mert95_pinned_model",
                       "MERTConfig": config_namespace["MERTConfig"]}
    suppressed = _exec_author_model(derived, model_namespace,
        cqt_disabled=json.loads(sources["config.json"]).get("feature_extractor_cqt") is False)
    return config_namespace["MERTConfig"], model_namespace["MERTModel"], suppressed


def _strict_state(model, state):
    """Explicit legacy weight-norm renaming, then exact keys/shapes/dtypes."""
    import torch

    expected = model.state_dict()
    if any(old not in state or new in state or new not in expected
           for old, new in WEIGHT_KEY_MAP.items()):
        raise MERTError("positional weight-norm key contract mismatch")
    mapped = {WEIGHT_KEY_MAP.get(k, k): v for k, v in state.items()}
    missing, extra = set(expected) - set(mapped), set(mapped) - set(expected)
    if missing or extra:
        raise MERTError(f"checkpoint keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for key, value in mapped.items():
        if (type(value) is not torch.Tensor or value.layout != torch.strided
                or value.device.type != "cpu" or value.dtype != torch.float32
                or value.shape != expected[key].shape or value.dtype != expected[key].dtype
                or not bool(torch.isfinite(value).all())):
            raise MERTError(f"checkpoint tensor contract mismatch: {key}")
    model.load_state_dict(mapped, strict=True, assign=True)
    if any(not torch.equal(value, mapped[key]) for key, value in model.state_dict().items()):
        raise MERTError("loaded checkpoint tensor equality failed")
    return {"tensors": len(mapped), "values": sum(v.numel() for v in mapped.values()),
            "key_mapping": dict(WEIGHT_KEY_MAP), "loaded_tensors_exact": True}


def load_model(root, device):
    """Return (author MERTModel, Wav2Vec2FeatureExtractor), offline and frozen.

    All failures propagate. No CPU fallback is substituted for requested CUDA.
    Only eager attention differs from the authenticated author configuration.
    Heavy imports and all model allocation occur here, never on module import.
    """
    import torch
    import transformers
    from transformers import Wav2Vec2FeatureExtractor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise MERTError(f"requires audited transformers {TRANSFORMERS_VERSION}")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise MERTError("MERT supports explicit CPU or CUDA only")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise MERTError("requested MERT CUDA is unavailable")
    root = Path(root).resolve()
    sources = _verified_sources(root)
    state = _checkpoint(root)
    config_class, model_class, suppressed = _author_classes(sources)
    config = config_class(**json.loads(sources["config.json"]))
    config._attn_implementation = "eager"
    # Meta initialization preserves host RNG and avoids an extra full model copy.
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = model_class(config)
    coverage = _strict_state(model, state)
    model.requires_grad_(False)
    model.eval()
    model.to(device=selected, dtype=torch.float32)
    processor = Wav2Vec2FeatureExtractor(**json.loads(sources["preprocessor_config.json"]))
    _validate_processor(processor)
    model.mert_metadata = {
        "repo_id": REPO_ID, "revision": REVISION, "license": LICENSE,
        "file_sha256": dict(FILE_SHA256), "derived_model_sha256": DERIVED_MODEL_SHA256,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "sample_rate": SAMPLE_RATE, "crop_samples": CROP_SAMPLES,
        "crop": "center-floor", "short_audio": "repeat-tile",
        "normalization": "author Wav2Vec2FeatureExtractor only",
        "attention": "eager", "pooling": "last_hidden_state temporal mean float32",
        "device": str(selected), "coverage": coverage,
        "author_import_audit": {"known_unused_cqt_prints_suppressed": suppressed,
                                "exact_suppressed_line": OPTIONAL_CQT_WARNING,
                                "unknown_stdout_and_warnings_preserved": True},
    }
    return model, processor


def prepare_waveform(array24000):
    """Mono float32 at 24kHz -> independent contiguous float32[120000].

    Validate the whole input, center crop with floor offset, or repeat short
    input. Silence and length-one input are valid. No amplitude normalization.
    Sample rate is the caller's contract; it cannot be inferred from an array.
    """
    import numpy as np

    if (not isinstance(array24000, np.ndarray) or array24000.dtype != np.float32
            or array24000.ndim != 1 or not array24000.size
            or not np.isfinite(array24000).all()):
        raise ValueError("MERT input must be nonempty finite mono float32 at 24000 Hz")
    if array24000.size < CROP_SAMPLES:
        result = np.tile(array24000, (CROP_SAMPLES + array24000.size - 1) // array24000.size)[:CROP_SAMPLES]
    else:
        start = (array24000.size - CROP_SAMPLES) // 2
        result = array24000[start:start + CROP_SAMPLES]
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _validate_processor(processor):
    expected = {"sampling_rate": SAMPLE_RATE, "do_normalize": True,
                "return_attention_mask": True, "feature_size": 1,
                "padding_side": "right", "padding_value": 0}
    if any(getattr(processor, k, None) != v for k, v in expected.items()):
        raise MERTError("MERT author processor contract mismatch")


@contextmanager
def deterministic_inference():
    """Scoped backend flags, shared by adapter and direct-author parity.

    These flags are process-wide while the context is active. Use the dedicated
    sequential prepass; do not overlap other model inference in this process.
    No CUDA initialization, global deterministic-algorithm toggle, or environment
    mutation is performed. The caller may separately bind its cuBLAS environment.
    """
    import torch

    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_float32_matmul_precision(precision)


def embed_waveform(array24000, model, processor, device):
    """Shared CPU/GPU inference boundary; returns owned float32[768]."""
    import numpy as np
    import torch

    _validate_processor(processor)
    prepared = prepare_waveform(array24000)
    selected = torch.device(device)
    parameter = next(model.parameters())
    if (model.training or parameter.dtype != torch.float32
            or parameter.device.type != selected.type
            or (selected.index is not None and parameter.device.index != selected.index)
            or any(p.requires_grad for p in model.parameters())):
        raise MERTError("MERT requires frozen eval float32 on the requested device")
    inputs = processor(prepared, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                       return_attention_mask=True)
    if set(inputs) != {"input_values", "attention_mask"}:
        raise MERTError("unexpected MERT processor outputs")
    values, mask = inputs["input_values"], inputs["attention_mask"]
    if (values.shape != (1, CROP_SAMPLES) or values.dtype != torch.float32
            or not bool(torch.isfinite(values).all()) or mask.shape != values.shape
            or mask.dtype not in (torch.int32, torch.int64) or not bool((mask == 1).all())):
        raise MERTError("invalid normalized MERT input or all-ones attention mask")
    inputs = {key: value.to(selected) for key, value in inputs.items()}
    with deterministic_inference(), torch.inference_mode(), torch.autocast(device_type=selected.type, enabled=False):
        hidden = model(**inputs, output_hidden_states=False,
                       output_attentions=False, return_dict=True).last_hidden_state
        if (hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1
                or hidden.shape[2] != EMBEDDING_SIZE or hidden.dtype != torch.float32
                or not bool(torch.isfinite(hidden).all())):
            raise MERTError("invalid MERT last_hidden_state")
        embedding = hidden.mean(dim=1, dtype=torch.float32)[0].cpu().numpy().copy()
    if embedding.shape != (EMBEDDING_SIZE,) or not np.isfinite(embedding).all():
        raise MERTError("invalid pooled MERT embedding")
    return embedding


def embed_audio(path, model, processor, device):
    """Decode raw mixture at 24kHz, then the exact frozen waveform boundary.

    Caller owns audio identity binding, per-file error policy, and model cleanup.
    No separation, peak/RMS normalization, file cache, or cross-file state.
    """
    import librosa
    import numpy as np

    audio, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                              dtype=np.float32, res_type="kaiser_fast")
    if rate != SAMPLE_RATE:
        raise MERTError("decoded MERT sample rate is not 24000 Hz")
    return embed_waveform(audio, model, processor, device)
