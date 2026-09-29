"""Single-stem Eliya inference using the unchanged authenticated CPU loader.

Vendoring contract: keep this file beside the exact eliya_forensics.py in a
package with __init__.py. No project package install or Hub access is required.
Upstream attribution/license: eliya Forensics base, CC-BY-NC-4.0.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from . import eliya_forensics as cpu

CPU_ADAPTER_SHA256 = "279ce521339495ca6cde144257ee1463bf0ad26c4ff59d56b7d32d47c5d31179"


def _requested_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Eliya runtime supports explicit cpu or cuda devices only")
    if target.type == "cpu":
        if target.index not in (None, 0):
            raise ValueError("CPU device index must be omitted or zero")
        return torch.device("cpu")
    return target


def _resolve_device(target: torch.device) -> torch.device:
    # Called only AFTER strict_load_cpu's CUDA-hiding guard has exited.
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        index = torch.cuda.current_device() if target.index is None else target.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index unavailable: {index}")
        return torch.device("cuda", index)
    return target


def _check_model(model: nn.Module, target: torch.device) -> None:
    if any(module.training for module in model.modules()):
        raise ValueError("Eliya model must be in eval mode")
    for name, value in chain(model.named_parameters(), model.named_buffers()):
        if value.device != target:
            raise ValueError(f"model tensor {name} is on {value.device}, requested {target}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"model tensor {name} must be float32")


def load_eliya(root: str | Path, device: str | torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Authenticate/strict-load on CPU, restore its guard, then transfer once.

    All authentication/architecture/transfer errors propagate. The returned
    receipt retains the full 651-tensor CPU audit and four classified extras.
    No retry, alternate checkpoint, random/pretrained fallback or installation.
    """
    requested = _requested_device(device)
    if cpu.sha256(Path(cpu.__file__)) != CPU_ADAPTER_SHA256:
        raise cpu.AuditError("CPU adapter source hash mismatch")
    model, cpu_receipt = cpu.strict_load_cpu(Path(root))
    target = _resolve_device(requested)
    started = perf_counter()
    model = model.to(target)
    _check_model(model, target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    receipt = {
        "root": str(Path(root).resolve()),
        "device_requested": str(requested), "device": str(target),
        "cpu_adapter_sha256": CPU_ADAPTER_SHA256,
        "runtime_sha256": cpu.sha256(Path(__file__)),
        "cpu_load": cpu_receipt,
        "transfer_seconds": perf_counter() - started,
        "dtype": "torch.float32", "fallback": False,
    }
    return model, receipt


def predict_eliya_fake(
    model: nn.Module,
    audio: npt.NDArray[np.float32] | Tensor,
    device: str | torch.device,
) -> float:
    """Return a scalar fake score for raw mono 16 kHz audio, without mutation.

    NumPy input must be 1-D float32; tensor input must also be on CPU. The caller
    owns decoding/mono conversion/resampling to 16 kHz. Preserve the author's
    peak normalization BEFORE centered 80,000-sample crop or repeat padding by
    delegating verbatim to prepare_waveform(audio, 16000). Silence stays zero.
    Invalid input/output or runtime failures raise; no substitute score is made.
    """
    if isinstance(audio, np.ndarray):
        if audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError("expected one-dimensional NumPy float32 mono 16 kHz audio")
        # Copy supports read-only/negative-stride views and avoids writable aliases.
        waveform = torch.from_numpy(audio.copy())
    elif isinstance(audio, Tensor):
        if audio.ndim != 1:
            raise ValueError("expected one-dimensional CPU float32 mono 16 kHz audio")
        waveform = audio
    else:
        raise TypeError("audio must be a NumPy float32 array or CPU float32 tensor")
    prepared = cpu.prepare_waveform(waveform, 16000)
    target = _resolve_device(_requested_device(device))
    _check_model(model, target)
    # Explicit float32 execution even if caller entered an autocast context.
    with torch.inference_mode(), torch.autocast(device_type=target.type, enabled=False):
        logits = model(prepared.unsqueeze(0).to(target))
        if (not isinstance(logits, Tensor) or logits.shape != (1,)
                or logits.device != target or logits.dtype != torch.float32):
            raise ValueError("expected one float32 logit shaped (1,) on requested device")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Eliya logit must be finite")
        fake = 1.0 - torch.sigmoid(logits).float()
        if not bool(torch.isfinite(fake).all()):
            raise ValueError("Eliya fake probability must be finite")
        return float(fake.item())
