"""Fixed first/center/end Eliya input adaptation for sub42.

Upstream: eliya/forensics_0.3B_base_deepfake_classifier, CC-BY-NC-4.0.
Adaptation: full-stem-normalized temporal windows and a fixed probability mean.
Standalone for vendoring/source embedding; no checkpoint loading or fallback.
The caller retains old39 VOICE on error and keeps original center E0 for FILE.
"""
from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn


WINDOW_SAMPLES = 80000


def _check_audio(audio: npt.NDArray[np.float32]) -> None:
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a NumPy float32 array")
    if audio.ndim != 1 or audio.dtype != np.float32 or audio.size == 0:
        raise ValueError("expected nonempty one-dimensional NumPy float32 mono 16 kHz audio")
    if not bool(np.isfinite(audio).all()):
        raise ValueError("audio must be finite")


def prepare_temporal(
    audio: npt.NDArray[np.float32],
) -> tuple[tuple[int, ...], int, tuple[Tensor, ...]]:
    """Return sorted unique starts, center start, and independent CPU windows.

    Short inputs have start zero and the author's repeat padding. Prediction
    bypasses this preparation for N <= 80000, reusing the original E0 directly.
    """
    _check_audio(audio)
    wav = torch.from_numpy(audio.copy())
    wav = wav / (wav.abs().max() + 1e-8)
    end = max(0, audio.size - WINDOW_SAMPLES)
    center_start = end // 2
    starts = tuple(sorted({0, center_start, end}))
    if audio.size < WINDOW_SAMPLES:
        wav = wav.repeat((WINDOW_SAMPLES + audio.size - 1) // audio.size)
    prepared = tuple(wav[start:start + WINDOW_SAMPLES].clone() for start in starts)
    if any(not bool(torch.isfinite(window).all()) for window in prepared):
        raise ValueError("prepared waveform must be finite")
    return starts, center_start, prepared


def _target_device(device: str | torch.device) -> torch.device:
    # These explicit-device guards mirror the pinned original Eliya runtime.
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Eliya runtime supports explicit cpu or cuda devices only")
    if target.type == "cpu":
        if target.index not in (None, 0):
            raise ValueError("CPU device index must be omitted or zero")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    index = torch.cuda.current_device() if target.index is None else target.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index unavailable: {index}")
    return torch.device("cuda", index)


def score_prepared(
    modelstate: dict[str, Any], prepared: Tensor, device: str | torch.device,
) -> float:
    """Score one already normalized CPU float32 window with the existing model.

    No normalization, loading, model transfer, retry, or substitute score occurs.
    Invalid contracts raise; forward errors reach the caller unchanged.
    """
    if (not isinstance(modelstate, dict)
            or not isinstance(modelstate.get("model"), nn.Module)
            or not callable(modelstate.get("predict"))
            or not isinstance(modelstate.get("receipt"), dict)):
        raise ValueError("Eliya model state requires model, predict, and receipt")
    if (not isinstance(prepared, Tensor) or prepared.shape != (WINDOW_SAMPLES,)
            or prepared.device.type != "cpu" or prepared.dtype != torch.float32):
        raise ValueError("expected prepared CPU float32 waveform shaped (80000,)")
    if not bool(torch.isfinite(prepared).all()):
        raise ValueError("prepared waveform must be finite")
    target = _target_device(device)
    model = modelstate["model"]
    if any(module.training for module in model.modules()):
        raise ValueError("Eliya model must be in eval mode")
    for name, value in chain(model.named_parameters(), model.named_buffers()):
        if value.device != target:
            raise ValueError(f"model tensor {name} is on {value.device}, requested {target}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"model tensor {name} must be float32")
    with torch.inference_mode(), torch.autocast(device_type=target.type, enabled=False):
        logits = model(prepared.detach().clone().unsqueeze(0).to(target))
        if (not isinstance(logits, Tensor) or logits.shape != (1,)
                or logits.device != target or logits.dtype != torch.float32):
            raise ValueError("expected one float32 logit shaped (1,) on requested device")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Eliya logit must be finite")
        fake = 1.0 - torch.sigmoid(logits).float()
        if not bool(torch.isfinite(fake).all()):
            raise ValueError("Eliya fake probability must be finite")
        return float(fake.item())


def _probability(value: Any) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (float, int, np.floating, np.integer))
            or not bool(np.isfinite(value)) or not 0.0 <= value <= 1.0):
        raise ValueError("center_fake must be a finite scalar probability in [0, 1]")
    return float(value)


def predict_temporal_fake(
    modelstate: dict[str, Any], audio: npt.NDArray[np.float32],
    center_fake: float, device: str | torch.device,
    evidence: dict[str, Any] | None = None,
) -> float:
    """Reuse E0 once and mean sorted native probabilities in NumPy float64.

    At most two extra forwards. Evidence, if supplied, must be a NEW dictionary,
    never old39 evidence; it is updated only after all windows succeed. Errors
    propagate for the caller's whole-VOICE fallback, never a partial mean.
    Short inputs return E0 without preparation or model/device access.
    """
    _check_audio(audio)
    center = _probability(center_fake)
    if evidence is not None and not isinstance(evidence, dict):
        raise TypeError("evidence must be a new dictionary or None")
    if audio.size <= WINDOW_SAMPLES:
        return center
    starts, center_start, prepared = prepare_temporal(audio)
    probabilities = tuple(
        center if start == center_start else score_prepared(modelstate, window, device)
        for start, window in zip(starts, prepared)
    )
    result = float(np.mean(sorted(probabilities), dtype=np.float64))
    if evidence is not None:
        evidence.update(starts=starts, center_start=center_start,
                        probabilities=probabilities, temporal_fake=result)
    return result
