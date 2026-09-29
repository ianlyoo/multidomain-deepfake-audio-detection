"""Exact local implementation of the DACON 236749 public metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

LABEL_COLUMNS = (
    "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
    "VOICE_PRESENT", "MUSIC_PRESENT",
)
PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)

@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    ads: float
    cps: float
    file_eer: float
    voice_eer: float
    music_eer: float
    voice_presence_auc: float
    music_presence_auc: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}

def _as_binary(name: str, values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    if np.unique(array).size != 2:
        raise ValueError(f"{name} must contain both classes")
    return array.astype(np.int8, copy=False)

def _as_scores(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} must be in [0, 1]")
    return array

def official_eer(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float:
    """Mirror DACON's roc_curve/nearest-crossing EER definition exactly."""
    labels = _as_binary("y_true", y_true)
    scores = _as_scores("y_score", y_score)
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)

def _column(table: Mapping[str, Sequence[float] | np.ndarray], name: str) -> np.ndarray:
    try:
        return np.asarray(table[name]).reshape(-1)
    except KeyError as exc:
        raise KeyError(f"missing required column: {name}") from exc

def official_score(
    labels: Mapping[str, Sequence[int] | np.ndarray],
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> ScoreBreakdown:
    """Calculate Score, ADS, CPS, and all five component metrics."""
    label_arrays = {name: _column(labels, name) for name in LABEL_COLUMNS}
    prediction_arrays = {name: _column(predictions, name) for name in PREDICTION_COLUMNS}
    lengths = {a.size for a in (*label_arrays.values(), *prediction_arrays.values())}
    if len(lengths) != 1:
        raise ValueError("all label and prediction columns must have equal length")

    file_eer = official_eer(label_arrays["FILE_FAKE"], prediction_arrays["FILE_FAKE_PROB"])
    voice_mask = label_arrays["VOICE_PRESENT"] == 1
    music_mask = label_arrays["MUSIC_PRESENT"] == 1
    voice_eer = official_eer(label_arrays["VOICE_FAKE"][voice_mask], prediction_arrays["VOICE_FAKE_PROB"][voice_mask])
    music_eer = official_eer(label_arrays["MUSIC_FAKE"][music_mask], prediction_arrays["MUSIC_FAKE_PROB"][music_mask])

    voice_present = _as_binary("VOICE_PRESENT", label_arrays["VOICE_PRESENT"])
    music_present = _as_binary("MUSIC_PRESENT", label_arrays["MUSIC_PRESENT"])
    voice_scores = _as_scores("VOICE_PRESENT_PROB", prediction_arrays["VOICE_PRESENT_PROB"])
    music_scores = _as_scores("MUSIC_PRESENT_PROB", prediction_arrays["MUSIC_PRESENT_PROB"])
    voice_auc = float(roc_auc_score(voice_present, voice_scores))
    music_auc = float(roc_auc_score(music_present, music_scores))

    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    cps = 0.5 * voice_auc + 0.5 * music_auc
    score = 0.9 * ads + 0.1 * cps
    return ScoreBreakdown(score, ads, cps, file_eer, voice_eer, music_eer, voice_auc, music_auc)
