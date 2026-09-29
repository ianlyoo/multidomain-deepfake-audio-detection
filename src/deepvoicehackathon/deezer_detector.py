"""Fixed-native-crop adapter for the public Deezer MUSIC research model.

Adapted from Deezer/deepfake-detector, edc94ad04b721e4ba59ccfb25e14606ddaa4a78a,
by D. Afchar, G. Meseguer-Brocal and R. Hennequin (ICASSP 2025), CC-BY-NC-4.0.
https://github.com/deezer/deepfake-detector/blob/edc94ad04b721e4ba59ccfb25e14606ddaa4a78a/LICENSE.md
Changes: CPU PyTorch layout conversion and strict fixed-crop validation only.
This is NOT Deezer's production detector. No file pooling/resampling/padding
policy or probability fitting is provided. TensorFlow is not a dependency.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 44100
CROP_SAMPLES = 34398
INPUT_SHAPE = (64, 743, 1)
CHANNELS = (16, 32, 64, 128, 256, 512)
HANN_PATH = Path(os.environ.get("DATA_DIR", "data")) / "auxiliary/author-exact-hann-f32.npy"
HANN_SHA256 = "0017068cbd40b050e87598bb5d16b41f958a54c1b2ff3ff61784b06a66cc8324"
HANN_DATA_SHA256 = "e689f7ccf093a825f8c6252e583e432911bd546aa2ae18c9381b891addb455ec"
# Exact TF2.15 float32 periodic Hann export, not a regenerated mathematical window.
# Provenance: deezer-frontend-diagnosis-20260912-v1/tf-reference-v1.json;
# authenticated author revision/license above; 2048 values / 8192 payload bytes.


def load_author_hann(path: Path = HANN_PATH) -> np.ndarray:
    """Read and authenticate the same bytes before parsing; never trust a stale cache."""
    with path.open("rb") as stream:
        data = stream.read(8321)
    if len(data) != 8320 or hashlib.sha256(data).hexdigest() != HANN_SHA256:
        raise ValueError("Author Hann file SHA256/size mismatch")
    window = np.load(io.BytesIO(data), allow_pickle=False)
    if window.dtype != np.dtype("<f4") or window.shape != (2048,):
        raise ValueError("Author Hann must contain exactly 2048 little-endian float32 values")
    if not np.isfinite(window).all() or hashlib.sha256(window.tobytes()).hexdigest() != HANN_DATA_SHA256:
        raise ValueError("Author Hann data SHA256/value mismatch")
    window.setflags(write=False)
    return window


def native_crop_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                         window_path: Path = HANN_PATH) -> torch.Tensor:
    """Exactly one mono [T] or channel-last stereo [T,2] crop -> [1,1,64,743].

    Reject other lengths/rates instead of inventing whole-file/short-file policy.
    Mono is equivalent to the author's channel duplication then 0.5/0.5 mix.
    """
    array = np.asarray(audio)
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Only native 44100 Hz crops are supported; no resampling policy")
    if array.shape not in ((CROP_SAMPLES,), (CROP_SAMPLES, 2)):
        raise ValueError(f"Expected [{CROP_SAMPLES}] or [{CROP_SAMPLES},2] fixed crop")
    if array.dtype != np.float32:
        raise TypeError("Audio must be float32")
    if not np.isfinite(array).all():
        raise ValueError("Audio contains nonfinite values")
    import scipy
    from scipy.fft import rfft
    if scipy.__version__ != "1.15.3":
        raise RuntimeError("Deezer frontend requires the audited SciPy 1.15.3 float32 FFT")
    window = load_author_hann(window_path)
    x = array.copy()
    if x.ndim == 2:
        x = np.float32(0.5) * x[:, 0] + np.float32(0.5) * x[:, 1]
    # No centering/end padding; preserve float32 products and complex64 FFT.
    frames = np.lib.stride_tricks.sliding_window_view(x, 2048)[::512]
    product = np.ascontiguousarray(frames * window)
    spectrum = rfft(product, n=2048, axis=-1, workers=2)
    if spectrum.dtype != np.complex64 or spectrum.shape != (64, 1025):
        raise RuntimeError("Unexpected SciPy FFT dtype/shape")
    power = np.clip(np.abs(spectrum) ** 2, 1e-10, 1e6)
    features = (np.log(power) / np.log(np.float32(10.0)) + 4.0) / 3.0
    features = np.ascontiguousarray(features[:, :743])
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Frontend produced invalid float32 features")
    return torch.from_numpy(features)[None, None]


def tensor_fingerprint(array: np.ndarray) -> str:
    """Canonical dtype/shape/content identity, independent of NPZ container metadata."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class DeezerSpecNN(nn.Module):
    """Inference-only six-block network; retains BOTH author output heads."""

    def __init__(self, encoder_classes: int = 10) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        previous = 1
        for channels in CHANNELS:
            self.convs.append(nn.Conv2d(previous, channels, 3, padding=1))
            self.norms.append(nn.BatchNorm2d(channels, eps=0.001))
            previous = channels
        self.dense = nn.Linear(512, 128)
        self.deepfake = nn.Linear(128, 1)
        self.encoder = nn.Linear(128, encoder_classes)
        self.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("Deezer readiness adapter supports inference only")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("CPU float32 features required")
        if tuple(x.shape) != (1, 1, 64, 743) or not torch.isfinite(x).all():
            raise ValueError("Expected finite features [1,1,64,743]")
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.max_pool2d(F.relu(conv(x)), 2, stride=2))
        x = F.silu(self.dense(x.mean(dim=(2, 3))))
        return torch.sigmoid(self.deepfake(x)), torch.softmax(self.encoder(x), dim=-1)

    @torch.inference_mode()
    def predict_fake_from_features(self, features: torch.Tensor) -> float:
        """Score one fixed crop's features; no whole-file scoring policy is implied."""
        real, _ = self(features)
        # AudioLoader.get_file_path and create_fast_eval_iterator explicitly REAL=1.
        return float(1.0 - real.item())


def weight_mapping() -> dict[str, tuple[str, tuple[int, ...] | None]]:
    mapping = {}
    for i in range(6):
        suffix = f"_{i}" if i else ""
        mapping[f"conv2d{suffix}/kernel:0"] = (f"convs.{i}.weight", (3, 2, 0, 1))
        mapping[f"conv2d{suffix}/bias:0"] = (f"convs.{i}.bias", None)
        for tf_name, pt_name in (("gamma", "weight"), ("beta", "bias"),
                                 ("moving_mean", "running_mean"), ("moving_variance", "running_var")):
            mapping[f"batch_normalization{suffix}/{tf_name}:0"] = (f"norms.{i}.{pt_name}", None)
    for name in ("dense", "deepfake", "encoder"):
        mapping[f"{name}/kernel:0"] = (f"{name}.weight", (1, 0))
        mapping[f"{name}/bias:0"] = (f"{name}.bias", None)
    return mapping


def load_converted(path: Path, *, expected_sha256: str) -> tuple[DeezerSpecNN, dict]:
    """Authenticate first, then exact all-tensor conversion; missing/extra fails closed."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("Converted weight SHA256 mismatch")
    mapping = weight_mapping()
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(mapping) or set(archive.files) != set(mapping):
            raise ValueError("Inference tensor inventory mismatch")
        encoder_shape = archive["encoder/bias:0"].shape
        if len(encoder_shape) != 1 or encoder_shape[0] < 2:
            raise ValueError("Invalid encoder head")
        model = DeezerSpecNN(encoder_shape[0])
        state = model.state_dict()
        records = []
        for author_name, (target_name, permutation) in mapping.items():
            source = archive[author_name]
            if source.dtype != np.float32 or not np.isfinite(source).all():
                raise ValueError(f"Invalid tensor dtype/value: {author_name}")
            converted = np.ascontiguousarray(source.transpose(permutation) if permutation else source)
            if tuple(converted.shape) != tuple(state[target_name].shape):
                raise ValueError(f"Tensor shape mismatch: {author_name}")
            # Round trip layout conversion must preserve every float32 bit.
            restored = converted.transpose(np.argsort(permutation)) if permutation else converted
            if tensor_fingerprint(restored) != tensor_fingerprint(source):
                raise ValueError(f"Nonlossless conversion: {author_name}")
            state[target_name] = torch.from_numpy(converted.copy())
            records.append({"author": author_name, "target": target_name, "shape": list(source.shape),
                            "source_fingerprint": tensor_fingerprint(source),
                            "converted_fingerprint": tensor_fingerprint(converted)})
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
    return model, {"source_tensor_count": len(records), "mapped_tensor_count": len(records),
                   "omitted_inference_tensors": [], "auxiliary_head": "encoder retained and validated",
                   "generated_state": [f"norms.{i}.num_batches_tracked=0 (unused in eval)" for i in range(6)],
                   "tensors": records}
