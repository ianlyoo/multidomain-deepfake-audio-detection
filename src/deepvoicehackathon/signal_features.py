"""Fixed per-file signal measurements; no fitted or batch-level statistics."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import expit

from .file_backend import EPSILON, FEATURES as BASE_FEATURES, features as base_features

SAMPLE_RATE = 16000
SIGNAL_FEATURES = (
    'signal:highband_power_fraction', 'signal:rolloff95_nyquist',
    'signal:spectral_flatness', 'signal:clipped_sample_fraction',
    'signal:silent_frame_fraction',
)
FEATURES = (*BASE_FEATURES, *SIGNAL_FEATURES)
DEFINITION = dict(
    version=1, sample_rate=SAMPLE_RATE, mono=True, dtype='float32',
    loader='librosa.load(sr=16000, mono=True, dtype=np.float32); default soxr_hq',
    fft_samples=512, hop_samples=256, window='periodic Hann',
    framing='start at zero; minimum frames covering file; zero-pad final frame; no centering',
    spectrum='float64 mean squared rFFT magnitude, DC excluded, Nyquist included',
    highband='sum power at f >= 4000 Hz / total non-DC power',
    rolloff='first bin cumulative power >= 0.95 total / 8000 Hz',
    flatness='geometric / arithmetic mean of mean power, floor=max_power*1e-12',
    clipping='fraction of actual samples with abs(x) >= 0.999',
    silence='fraction of nonoverlapping 320-sample frames with RMS <= 0.001; actual tail length',
    zero_power='highband, rolloff, flatness = 0',
    invalid='reject empty, non-mono or nonfinite input; finite outputs in [0,1]',
    feature_names=list(SIGNAL_FEATURES),
)
DEFINITION_SHA256 = hashlib.sha256(
    json.dumps(DEFINITION, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()


def load_audio(audio_path):
    """Same load and resampling call as the incumbent, without importing models."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Invalid audio: {audio_path}')
    return audio


def extract(audio):
    """Measure one already decoded mono 16 kHz waveform in fixed feature order."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('nonempty finite mono 16 kHz waveform required')
    clipping = float(np.mean(np.abs(x) >= .999))
    # Scaling avoids overflow and does not affect the three relative spectra.
    peak = np.max(np.abs(x))
    normalized = x / peak if peak else x
    n_frames = 1 + max(0, (len(x) - 512 + 255) // 256)
    padded = np.pad(normalized, (0, (n_frames - 1) * 256 + 512 - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 512)[::256]
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(512) / 512)
    power = np.mean(np.abs(np.fft.rfft(frames * window)) ** 2, axis=0)[1:]
    total = float(power.sum())
    highband = rolloff = flatness = 0.
    if total > 0:
        frequency = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)[1:]
        highband = float(power[frequency >= 4000].sum() / total)
        index = min(int(np.searchsorted(np.cumsum(power), .95 * total)), len(power) - 1)
        rolloff = float(frequency[index] / (SAMPLE_RATE / 2))
        bounded = np.maximum(power, power.max() * 1e-12)
        flatness = float(np.exp(np.mean(np.log(bounded))) / bounded.mean())
    silent = []
    for start in range(0, len(x), 320):
        frame = x[start:start + 320]
        frame_peak = np.max(np.abs(frame))
        # Stable RMS without squaring large finite samples.
        rms = frame_peak * np.sqrt(np.mean((frame / frame_peak) ** 2)) if frame_peak else 0.
        silent.append(rms <= .001)
    result = np.array([highband, rolloff, flatness, clipping, np.mean(silent)])
    if not np.isfinite(result).all():
        raise ValueError('nonfinite signal features')
    return np.clip(result, 0., 1.)


def features(rows):
    original = base_features(rows)
    try:
        signal = np.asarray([[float(row[name]) for name in SIGNAL_FEATURES] for row in rows])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('missing/invalid signal feature input') from error
    if not np.isfinite(signal).all() or ((signal < 0) | (signal > 1)).any():
        raise ValueError('signal features must be finite in [0,1]')
    return np.column_stack((original, signal))


def predict(model, rows):
    if (model.get('version') != 6 or model.get('model_type') != 'gradient_boosted_trees_signal22'
            or model.get('feature_names') != list(FEATURES) or model.get('epsilon') != EPSILON
            or model.get('input_dtype') != 'float32'
            or model.get('signal_definition_sha256') != DEFINITION_SHA256):
        raise ValueError('incompatible signal feature export')
    x = features(rows).astype(np.float32)
    raw = np.full(len(rows), model['initial_log_odds'], dtype=np.float64)
    for tree in model['trees']:
        for i, values in enumerate(x):
            node = 0
            while tree['children_left'][node] != -1:
                node = (tree['children_left'][node]
                        if float(values[tree['feature'][node]]) <= float(tree['threshold'][node])
                        else tree['children_right'][node])
            raw[i] += model['learning_rate'] * tree['value'][node]
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite tree log odds')
    return expit(raw)
