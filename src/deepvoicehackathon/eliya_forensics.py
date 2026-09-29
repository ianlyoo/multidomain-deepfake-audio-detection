"""Pinned, CPU-only feasibility adapter for eliya's WavLM/AASIST detector.

The reviewed upstream architecture is hash-bound and instantiated on meta first.
No Hub calls, pickle fallback, checkpoint key guessing, or scoring integration.
CC-BY-NC-4.0 upstream attribution: eliya/forensics_0.3B_base_deepfake_classifier.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator
from unittest.mock import patch

import torch
from torch import Tensor, nn

REPO_ID = "eliya/forensics_0.3B_base_deepfake_classifier"
REVISION = "dc4d36f1846b1ac190c1f24c8dc034fa32f9f2d6"
BACKBONE_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
WEIGHT_FILE = "checkpoint_epoch_5.safetensors"
WEIGHT_SIZE = 1269932956
WEIGHT_SHA256 = "803d2c57e705395d007b9deaf5b1a110965950f207df223e6dff5b6d298490fb"
SOURCE_SHA256 = {
    "model.py": "c99898e4f8fd5fb9ac1edfbc8736033d0a21680ccb58541e23c4188a36c860c9",
    "inference.py": "53f16ec69e325ed479ea51de83579b2c1ee912a6de1d27f3fb5bef735ee66034",
    "config.json": "c0707e960fce86299863f317fcd9b6f16f02cfa1fd8a40b9a0ec0c3ffccd82e9",
    "requirements.txt": "bd83aab21f1f47541406aeab4738054257f6656f5ed533fb02be1e8978837f99",
    "README.md": "99a1e755a7add2c4babeedcd52aca7bd7fbf8666ab178976ff2649125c13fd01",
    "backbone/config.json": "a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d",
}
TRAINING_EXTRA_SHAPES = {
    "projection.0.bias": (320,), "projection.0.weight": (320, 320),
    "projection.2.bias": (320,), "projection.2.weight": (320, 320),
}


class AuditError(RuntimeError):
    """Fail-closed provenance, architecture, or checkpoint contract failure."""


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def offline_audit() -> Iterator[None]:
    """Process-local network tripwire; use in an isolated audit process only.

    Reviewed code uses Python networking only. This is not an OS sandbox for
    arbitrary untrusted native code. Environment/patches are restored on exit.
    """
    def blocked(*args, **kwargs):
        raise AuditError("network access blocked by offline CPU audit")

    with ExitStack() as stack:
        stack.enter_context(patch.dict("os.environ", {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "CUDA_VISIBLE_DEVICES": "",
        }))
        for target in ("socket.create_connection", "socket.getaddrinfo",
                       "socket.socket.connect", "socket.socket.connect_ex",
                       "socket.socket.sendto"):
            stack.enter_context(patch(target, blocked))
        yield


def verify_sources(root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if path.stat().st_size > 1_000_000:
            raise AuditError(f"source/config exceeds 1 MB: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise AuditError(f"source hash mismatch: {relative}")
    return actual


def _architecture(root: Path) -> nn.Module:
    """Execute only the reviewed, pinned model source with one AST substitution.

    Call only after verify_sources. Meta construction audits the complete shape
    contract before any real parameter is allocated/loaded.
    """
    from transformers import WavLMConfig

    tree = ast.parse((root / "model.py").read_text(encoding="utf-8"))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value) == "WavLMModel.from_pretrained('microsoft/wavlm-large')"):
            node.value = ast.Call(func=ast.Name(id="WavLMModel", ctx=ast.Load()),
                                  args=[ast.Name(id="_offline_config", ctx=ast.Load())], keywords=[])
            replacements += 1
    if replacements != 1:
        raise AuditError(f"expected exactly one reviewed constructor, got {replacements}")
    config = WavLMConfig.from_dict(json.loads((root / "backbone/config.json").read_text()))
    namespace = {"__name__": "_eliya_reviewed_model", "_offline_config": config}
    exec(compile(ast.fix_missing_locations(tree), str(root / "model.py"), "exec"), namespace)
    with torch.device("meta"):
        return namespace["DeepfakeDetector"]()


def audit_state(model: nn.Module, state: dict[str, Tensor]) -> dict:
    """Require exact names, shapes and dtypes; enumerate every mismatch."""
    expected = model.state_dict()
    shared = expected.keys() & state.keys()
    findings = {
        "missing": sorted(expected.keys() - state.keys()),
        "unexpected": sorted(state.keys() - expected.keys()),
        "shape_mismatches": {
            k: {"expected": list(expected[k].shape), "actual": list(state[k].shape)}
            for k in sorted(shared) if expected[k].shape != state[k].shape
        },
        "dtype_mismatches": {
            k: {"expected": str(expected[k].dtype), "actual": str(state[k].dtype)}
            for k in sorted(shared) if expected[k].dtype != state[k].dtype
        },
    }
    if any(findings.values()):
        raise AuditError("checkpoint contract mismatch: " + json.dumps(findings, sort_keys=True))
    nonfinite = [k for k, v in state.items() if not bool(torch.isfinite(v).all())]
    if nonfinite:
        raise AuditError("nonfinite checkpoint tensors: " + json.dumps(sorted(nonfinite)))
    return findings


def split_training_extras(state: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict]:
    """Account for the pinned checkpoint's four inference-unreachable tensors.

    Published model.py never defines/references projection. The exact original
    training projection architecture/purpose is not shipped; contrastive-head
    attribution is an inference, not an independently verified training claim.
    Unknown extras still fail audit_state. Tensor storage is not copied.
    """
    classified = {}
    for key, shape in TRAINING_EXTRA_SHAPES.items():
        value = state.get(key)
        if (value is None or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())):
            raise AuditError(f"training-extra contract mismatch: {key}")
        classified[key] = {
            "shape": list(shape), "dtype": str(value.dtype), "finite": True,
            "tensor_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "classification": "auxiliary projection absent from published inference graph",
        }
    return {k: v for k, v in state.items() if k not in classified}, classified


def strict_load_cpu(root: Path) -> tuple[nn.Module, dict]:
    """Authenticate, meta-audit, then strict-load all inference parameters on CPU."""
    from safetensors.torch import load_file

    root = Path(root).resolve()
    started = perf_counter()
    sources = verify_sources(root)
    path = root / WEIGHT_FILE
    if path.stat().st_size != WEIGHT_SIZE or sha256(path) != WEIGHT_SHA256:
        raise AuditError("checkpoint size/SHA-256 mismatch")
    with offline_audit():
        model = _architecture(root)
        complete_state = load_file(str(path), device="cpu")
        state, training_extras = split_training_extras(complete_state)
        findings = audit_state(model, state)
        load_start = perf_counter()
        result = model.load_state_dict(state, strict=True, assign=True)
        load_seconds = perf_counter() - load_start
        if result.missing_keys or result.unexpected_keys:
            raise AuditError("strict load unexpectedly returned incompatible keys")
        loaded = model.state_dict()
        unequal = [k for k, v in loaded.items() if v.device.type != "cpu"
                   or not torch.equal(v, state[k])]
        if unequal:
            raise AuditError("loaded tensor mismatch: " + json.dumps(unequal))
        model.eval().requires_grad_(False)
        report = {
            "revision": REVISION, "backbone_revision": BACKBONE_REVISION,
            "checkpoint_sha256": WEIGHT_SHA256, "checkpoint_bytes": WEIGHT_SIZE,
            "source_sha256": sources, **findings,
            "checkpoint_tensors": len(complete_state),
            "state_tensors": len(state), "equal_loaded_tensors": len(loaded),
            "classified_training_extras": training_extras,
            "wavlm_tensors": sum(k.startswith("wavlm.") for k in state),
            "parameters": sum(p.numel() for p in model.parameters()),
            "all_finite": True, "all_parameters_cpu": True,
            "strict_load_seconds": load_seconds,
            "authenticated_load_audit_seconds": perf_counter() - started,
        }
    return model, report


def prepare_waveform(waveform: Tensor, sample_rate: int) -> Tensor:
    """Author's mono/resample/peaknorm/center-or-repeat recipe, CPU float32.

    Accept (T,) or channel-first (C,T); preserve all-zero silence. Empty and
    nonfinite inputs are errors, not fabricated waveforms or scores. Normalizing
    before crop intentionally preserves the author's full-file peak dependence.
    """
    if (waveform.device.type != "cpu" or waveform.dtype != torch.float32
            or waveform.ndim not in (1, 2) or waveform.numel() == 0):
        raise ValueError("expected nonempty CPU float32 waveform shaped (T,) or (C,T)")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite")
    wav = waveform.detach()
    if wav.ndim == 2:
        wav = wav.mean(0)
    if sample_rate != 16000:
        from torchaudio.functional import resample
        wav = resample(wav, sample_rate, 16000)
    if not bool(torch.isfinite(wav).all()):
        raise ValueError("mono/resampled waveform must be finite")
    wav = wav / (wav.abs().max() + 1e-8)
    n, cur = 80000, wav.shape[0]
    if cur < n:
        wav = wav.repeat((n + cur - 1) // cur)[:n]
    elif cur > n:
        start = (cur - n) // 2
        wav = wav[start:start + n]
    if not bool(torch.isfinite(wav).all()):
        raise ValueError("prepared waveform must be finite")
    return wav.contiguous()


def fake_probability(logits: Tensor) -> Tensor:
    if logits.device.type != "cpu" or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("expected finite floating-point CPU logits")
    return 1.0 - torch.sigmoid(logits).float()
