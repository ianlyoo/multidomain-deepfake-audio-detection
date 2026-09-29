"""Frozen same-weight CPU margin contract; no audio, training, or model loading."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL_SHA256 = "d7b5367bcf1e8247665835562524c8cbb8135ddd142540da4efd855bb55ebdff"
SOURCE_SHA256 = "0217a3cf65348e7bfffba87114598827978ddaa20ae7dd3f84bf7c2436ec1ff5"
ONNX_SHA256 = "af7a75c6ed457bc5b6941c8bc76aa06a66d48de40db944b761ed2bebfc0fbbd3"
NPZ_SHA256 = "d0bc3877a5d575bbe46a97794a72e463fd772258c395a2cbf3d4117aec72a29c"
W_SHA256 = "55af49df19e2e7091c9bbfd876a89332e4f5a4a721d346df334e2c9996b006a1"
B_SHA256 = "8c240bbb7f7327e0113d10980746ffa885df314385d4c2f13d000735f80520cb"
CONTRACT = {
    "version": 1,
    "feature_dim": 3585,
    "stored_dtype": "float32",
    "compute_dtype": "float64",
    "formula": "z64=dot(float64(fakeprint),float64(W[0]))+float64(b[0]);Q=0.5+atan(z64)/pi",
    "orientation": "higher_is_fake",
    "protocol_sha256": PROTOCOL_SHA256,
    "source_sha256": SOURCE_SHA256,
    "original_npz_sha256": NPZ_SHA256,
    "original_onnx_sha256": ONNX_SHA256,
    "weight_initializer_sha256": W_SHA256,
    "bias_initializer_sha256": B_SHA256,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


@dataclass(frozen=True)
class MarginWeights:
    weights: np.ndarray
    bias: float

    def score(self, fakeprint: np.ndarray) -> tuple[float, float]:
        x = np.asarray(fakeprint)
        if x.dtype != np.float32 or x.shape != (3585,):
            raise ValueError("Expected one float32 fakeprint of shape (3585,)")
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Fakeprint must be finite and in [0,1]")
        z64 = float(np.dot(x.astype(np.float64), self.weights) + self.bias)
        q = 0.5 + math.atan(z64) / math.pi
        if not math.isfinite(z64) or not 0.0 < q < 1.0:
            raise ValueError("Invalid margin result")
        return z64, q


def load_margin_weights(path: Path, expected_sha256: str | None = None) -> MarginWeights:
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Margin JSON SHA-256 mismatch")
    data = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(data, dict) or set(data) != set(CONTRACT) | {"weights", "bias"}:
        raise ValueError("Margin JSON keys differ from strict contract")
    for key, expected in CONTRACT.items():
        if type(data[key]) is not type(expected) or data[key] != expected:
            raise ValueError(f"Margin contract mismatch: {key}")
    arrays = []
    for key, shape, pin in (("weights", (1, 3585), W_SHA256), ("bias", (1,), B_SHA256)):
        # JSON numeric strings and bools must not be coerced into weights.
        raw = np.asarray(data[key], dtype=object)
        if raw.shape != shape or any(type(v) not in (int, float) for v in raw.flat):
            raise ValueError(f"Invalid {key} numeric shape/type")
        values = np.asarray(data[key], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            native = values.astype("<f4")
        if not np.isfinite(values).all() or not np.array_equal(native.astype(np.float64), values):
            raise ValueError(f"{key} must contain exact finite native float32 values")
        if hashlib.sha256(native.tobytes(order="C")).hexdigest() != pin:
            raise ValueError(f"{key} initializer SHA-256 mismatch")
        arrays.append(native)
    promoted = arrays[0][0].astype(np.float64)
    promoted.setflags(write=False)
    return MarginWeights(promoted, float(arrays[1][0]))


def score_with_margin(session, fakeprint: np.ndarray, parameters: MarginWeights | None):
    """Old ONNX failure propagates; auxiliary failure preserves L and is observable."""
    output = session.run(None, {"fakeprint": fakeprint.reshape(1, -1)})
    old_l = float(np.asarray(output[0]).reshape(-1)[0])
    if not np.isfinite(old_l) or not 0.0 <= old_l <= 1.0:
        raise ValueError(f"Unexpected lofcz score: {old_l}")
    try:
        if parameters is None:
            raise ValueError("Margin weights unavailable")
        z64, q = parameters.score(fakeprint)
        return old_l, z64, q, None
    except Exception as exc:
        return old_l, None, None, f"{type(exc).__name__}: {exc}"
