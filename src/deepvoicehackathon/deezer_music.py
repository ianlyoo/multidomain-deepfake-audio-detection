"""Frozen sub40 whole-file policy for the authenticated Deezer research adapter.

Failures deliberately propagate: the submission caller retains incumbent MUSIC.
No prediction cache, global runtime configuration, or TensorFlow is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import MutableMapping

import numpy as np
import torch
import torchaudio

from .deezer_detector import (
    CROP_SAMPLES, SAMPLE_RATE, DeezerSpecNN, load_author_hann,
    load_converted, native_crop_features,
)

WEIGHTS_SHA256 = "c1a71137ae4c351fc33e438d14cc484d2e1d67b34b32ff598d3d0f6c06167850"
MAX_SECONDS = 60


@dataclass(frozen=True, slots=True, eq=False)
class MusicModelContext:
    """Fixed context; the private, frozen-parameter model is inference-only.

    As with ordinary PyTorch modules, deliberate access to private tensor state
    can mutate it. Callers should retain this context and use predict_music_fake.
    The adapter authenticates window bytes again for every crop.
    """

    _model: DeezerSpecNN
    window_path: Path


def load_music_model(weights_path: str | Path,
                     window_path: str | Path) -> MusicModelContext:
    """Authenticate pinned assets and construct a CPU model without RNG drift."""
    weights = Path(weights_path).resolve()
    window = Path(window_path).resolve()
    load_author_hann(window)
    # Module initialization consumes CPU RNG even though all weights are loaded.
    # devices=[] avoids touching CUDA, including lazy initialization.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model, _ = load_converted(weights, expected_sha256=WEIGHTS_SHA256)
        model.float().eval()
        model.requires_grad_(False)
    return MusicModelContext(model, window)


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def prepare_audio(waveform: torch.Tensor | np.ndarray,
                  sample_rate: int) -> np.ndarray:
    """Channel-first CPU float32 -> mono float32 44100 Hz, without mutation.

    Validate the entire decoded input, retain first 60 native-rate seconds,
    average channels in float32, then resample once using torchaudio defaults.
    The returned array owns its storage; short-file tiling belongs to prediction.
    """
    rate = _positive_integer(sample_rate, "sample_rate")
    if isinstance(waveform, torch.Tensor):
        if waveform.device.type != "cpu" or waveform.dtype != torch.float32:
            raise TypeError("waveform must be CPU float32")
        array = waveform.detach().numpy()
    elif isinstance(waveform, np.ndarray):
        array = waveform
    else:
        raise TypeError("waveform must be a channel-first float32 tensor or ndarray")
    if array.dtype != np.float32:
        raise TypeError("waveform must be float32")
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("waveform must be nonempty channel-first [channels, samples]")
    if not np.isfinite(array).all():
        raise ValueError("waveform contains nonfinite values")
    mono = array[:, :MAX_SECONDS * rate].mean(axis=0, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("channel mean contains nonfinite values")
    if rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(
            torch.from_numpy(mono), rate, SAMPLE_RATE,
        ).detach().numpy()
    if mono.ndim != 1 or not mono.size or mono.dtype != np.float32 or not np.isfinite(mono).all():
        raise ValueError("resampling produced invalid mono float32 audio")
    return np.array(mono, dtype=np.float32, order="C", copy=True)


def crop_starts(sample_count: int) -> tuple[int, ...]:
    """Unique floor(i * (N - 34398) / 4) starts; short audio has one tiled crop."""
    count = _positive_integer(sample_count, "sample_count")
    span = max(0, count - CROP_SAMPLES)
    return tuple(dict.fromkeys(i * span // 4 for i in range(5)))


def _probability(value: float, label: str) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return score


def predict_music_fake(audio_path: str | Path, context: MusicModelContext,
                       evidence: MutableMapping[str, object] | None = None) -> float:
    """Decode the original file and return arithmetic mean P(fake) over crops.

    Optional evidence is updated only after all scores succeed. It contains
    decoded_sample_rate, decoded_channels, decoded_samples,
    retained_native_samples, resampled_samples, crop_starts, crops_44100
    (tuple of independent read-only float32 arrays), crop_scores, and mean_fake.
    Load/decode/frontend/model errors propagate unchanged to the caller.
    """
    waveform, rate = torchaudio.load(str(audio_path))
    mono = prepare_audio(waveform, rate)
    resampled_samples = int(mono.size)
    starts = crop_starts(mono.size)
    if mono.size < CROP_SAMPLES:
        mono = np.tile(mono, (CROP_SAMPLES + mono.size - 1) // mono.size)[:CROP_SAMPLES]
    scores = []
    captured = []
    with torch.inference_mode():
        for start in starts:
            crop = mono[start:start + CROP_SAMPLES]
            features = native_crop_features(crop, window_path=context.window_path)
            scores.append(_probability(context._model.predict_fake_from_features(features), "crop score"))
            if evidence is not None:
                saved = crop.copy()
                saved.setflags(write=False)
                captured.append(saved)
    mean = _probability(sum(scores) / len(scores), "aggregate score")
    if evidence is not None:
        # Report pre-tiling resampled size, using the actual prepared output.
        # Short crops may contain more samples than the resampled source.
        evidence.update({
            "decoded_sample_rate": int(rate),
            "decoded_channels": int(waveform.shape[0]),
            "decoded_samples": int(waveform.shape[1]),
            "retained_native_samples": min(int(waveform.shape[1]), MAX_SECONDS * int(rate)),
            "resampled_samples": resampled_samples,
            "crop_starts": starts,
            "crops_44100": tuple(captured),
            "crop_scores": tuple(scores),
            "mean_fake": mean,
        })
    return mean
