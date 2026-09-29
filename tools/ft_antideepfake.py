#!/usr/bin/env python3
"""Fine-tune the NII AntiDeepfake wav2vec2-style detector on a labeled manifest.

The model graph is the deployed fairseq-free frontend
(``src/deepvoicehackathon/_antideepfake_frontend.py``, byte-identical to the
sub55 package ``native_frontend.py``); it is imported, not copied, so a
zero-step checkpoint scores exactly like the deployed detector. Output index 0
is the FAKE class (model card: <fake, real>), matching ``predict_nii_fake``.

Manifest CSV (one or more ``--manifest``): ``path,label,group[,weight]``.
Column fallbacks: path <- relative_path|file|audio_path, label <- voice_fake|
file_fake, group <- speaker_id|speaker|content_id. Labels real/bonafide/0 and
fake/spoof/1. Relative paths resolve against the manifest directory.

Training reads random 4 s crops (repeat-padded when short) from the raw files
or, with ``--stems DIR``, from cached Demucs vocal stems ``DIR/<sha256>.npy``
written by ``tools/ft_antideepfake_score.py --vocal-stem --stem-cache DIR``.

Usage (GPU shared: default hard cap 8 GB via set_per_process_memory_fraction):
  .venv/Scripts/python.exe tools/ft_antideepfake.py --manifest M.csv --out $DATA_DIR/run1
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SR = 16_000
FAKE_INDEX = 0  # AntiDeepfake logits are <fake, real>
NII_MIN_AUDIO_SAMPLES = 400
NII_MAX_AUDIO_SAMPLES = 30 * SR
FRONTEND_PATH = ROOT / "src/deepvoicehackathon/_antideepfake_frontend.py"
FRONTEND_SHA256 = "80bc3a2375b6efaefb4cd5ad8e4d475ce1796c1d64beb178615d70328d38a744"
BASES = {
    "wav2vec-large": dict(
        path=Path(os.environ.get("DATA_DIR", "data")) / "models/nii_antideepfake/model.safetensors",
        sha256="b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5"),
    "mms300m": dict(
        path=Path(os.environ.get("DATA_DIR", "data")) / "models/nii-mms300m-antideepfake/model.safetensors",
        sha256="9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224"),
}
DEFAULT_STEM_DIR = Path(os.environ.get("DATA_DIR", "data")) / "features/vocal-stems"
AUDIO_EXT = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
LABELS = {"fake": 1, "spoof": 1, "1": 1, "real": 0, "bonafide": 0, "bona-fide": 0,
          "bona_fide": 0, "0": 0}
PATH_COLS = ("path", "relative_path", "file", "audio_path")
LABEL_COLS = ("label", "voice_fake", "file_fake")
GROUP_COLS = ("group", "speaker_id", "speaker", "content_id")


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def _pick(columns, candidates, what, required=True):
    for name in candidates:
        if name in columns:
            return name
    if required:
        raise ValueError(f"manifest lacks a {what} column (tried {candidates})")
    return None


def parse_label(value):
    key = str(value).strip().lower()
    if key not in LABELS:
        raise ValueError(f"unknown label {value!r}")
    return LABELS[key]


def read_manifest(path, root=None, tag=None):
    """Return rows {path, label, group, weight, sha256}; groups are prefixed by tag."""
    path = Path(path)
    root = Path(root) if root else path.parent
    tag = path.stem if tag is None else tag
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        pcol = _pick(columns, PATH_COLS, "path")
        lcol = _pick(columns, LABEL_COLS, "label")
        gcol = _pick(columns, GROUP_COLS, "group", required=False)
        rows = []
        for index, row in enumerate(reader):
            audio = Path(row[pcol])
            if not audio.is_absolute():
                audio = root / audio
            weight = float(row["weight"]) if row.get("weight") not in (None, "") else 1.0
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid weight in row {index}: {row.get('weight')!r}")
            group = row[gcol] if gcol and row.get(gcol) not in (None, "") else audio.stem
            rows.append(dict(path=str(audio), label=parse_label(row[lcol]),
                             group=f"{tag}:{group}", weight=weight,
                             sha256=(row.get("sha256") or "").strip().lower() or None,
                             meta={k: v for k, v in row.items()
                                   if k in ("system", "channel", "source")}))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def subsample(rows, limit, seed):
    """Label-stratified deterministic subsample (for smokes)."""
    if not limit or limit >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        members = [r for r in rows if r["label"] == label]
        take = min(len(members), int(round(limit * len(members) / len(rows))))
        out += [members[i] for i in sorted(rng.choice(len(members), take, replace=False))]
    return out


def group_split(rows, val_fraction, seed, attempts=200):
    """Speaker/group-disjoint split; val must contain both classes."""
    groups = sorted({r["group"] for r in rows})
    if len(groups) < 2:
        raise ValueError("group split needs at least two groups")
    rng = np.random.default_rng(seed)
    target = max(1, int(round(val_fraction * len(rows))))
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for _ in range(attempts):
        order = [groups[i] for i in rng.permutation(len(groups))]
        val_groups, count = set(), 0
        for g in order:
            if count >= target or len(val_groups) == len(groups) - 1:
                break
            val_groups.add(g)
            count += len(by_group[g])
        val = [r for r in rows if r["group"] in val_groups]
        train = [r for r in rows if r["group"] not in val_groups]
        if {r["label"] for r in val} == {0, 1} and {r["label"] for r in train} == {0, 1}:
            return train, val
    raise ValueError("could not find a group split with both classes on each side")


# ----------------------------------------------------------------------------
# Audio I/O, crops, stems
# ----------------------------------------------------------------------------

def load_audio(path):
    """Deployed decode: librosa.load(sr=16k, mono, float32) as script_sub55.load_audio."""
    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio


def stem_path(stem_dir, sha):
    return Path(stem_dir) / f"{sha}.npy"


def fit_length(audio, n, rng=None, start=None):
    """Random (or fixed-start) crop to n samples; short input is tiled (repeat)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("empty audio")
    if audio.size < n:
        audio = np.tile(audio, n // audio.size + 1)
    last = audio.size - n
    if start is None:
        start = int(rng.integers(0, last + 1)) if rng is not None else 0
    start = min(max(0, start), last)
    return audio[start:start + n].copy()


def trim_silence(audio, top_db):
    """Strip leading/trailing silence (librosa.effects.trim, top_db below peak).

    Applied identically to train and val so clip padding/length cannot act as a
    label shortcut; returns the input unchanged when trimming leaves < 0.25 s."""
    if not top_db:
        return audio
    import librosa
    trimmed, _ = librosa.effects.trim(np.asarray(audio, dtype=np.float32), top_db=top_db,
                                      frame_length=1024, hop_length=256)
    return trimmed if trimmed.size >= SR // 4 else audio


def read_crop(path, seconds, rng, stem_file=None, trim_db=0.0):
    """Read a 16 kHz mono crop, touching only the needed span when possible."""
    n = int(round(seconds * SR))
    if stem_file is not None:
        stem = np.load(stem_file, mmap_mode=None if trim_db else "r")
        return fit_length(trim_silence(stem, trim_db), n, rng)
    if trim_db:
        return fit_length(trim_silence(load_audio(path), trim_db), n, rng)
    import soundfile as sf
    try:
        info = sf.info(str(path))
        margin = int(info.samplerate * 0.05)
        need = int(math.ceil(seconds * info.samplerate)) + 2 * margin
        start = 0
        if info.frames > need:
            start = int(rng.integers(0, info.frames - need + 1)) if rng is not None else 0
            data, sr = sf.read(str(path), start=start, frames=need, dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        if audio.size == 0:
            raise ValueError("empty read")
    except Exception:
        audio = load_audio(path)
    return fit_length(audio, n, rng)


def normalize(waveform):
    """Exact predict_nii_fake per-utterance normalization (float64 stats)."""
    values = np.asarray(waveform, dtype=np.float32).astype(np.float64, copy=False)
    return ((values - values.mean()) / np.sqrt(values.var() + 1e-5)).astype(np.float32)


# ----------------------------------------------------------------------------
# Augmentation (MP3/telephone reuse tools/build_channel_probe.py settings)
# ----------------------------------------------------------------------------

@dataclass
class AugConfig:
    p_gain: float = 0.0
    gain_db: tuple = (-6.0, 6.0)
    p_noise: float = 0.0
    noise_snr_db: tuple = (10.0, 40.0)
    p_mp3: float = 0.0
    mp3_kbps: tuple = (32, 48, 64, 96, 128)
    p_telephone: float = 0.0
    p_music: float = 0.0
    music_snr_db: tuple = (0.0, 20.0)
    music_files: list = field(default_factory=list)

    def active(self):
        return any(p > 0 for p in (self.p_gain, self.p_noise, self.p_mp3, self.p_telephone,
                                   self.p_music))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(signal, other, snr_db):
    scale = _rms(signal) / (_rms(other) * 10.0 ** (snr_db / 20.0))
    return (signal + scale * other).astype(np.float32)


def add_noise(audio, snr_db, rng):
    return mix_at_snr(audio, rng.standard_normal(audio.size).astype(np.float32), snr_db)


def apply_gain(audio, gain_db):
    return np.clip(audio * 10.0 ** (gain_db / 20.0), -1.0, 1.0).astype(np.float32)


def mp3_roundtrip(audio, kbps=64):
    """lameenc encode (build_channel_probe.encode_mp3 settings, variable bitrate) + decode."""
    import lameenc
    import soundfile as sf
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(kbps))
    encoder.set_in_sample_rate(SR)
    encoder.set_out_sample_rate(SR)
    encoder.set_channels(1)
    encoder.set_quality(2)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    raw = encoder.encode(pcm.tobytes()) + encoder.flush()
    decoded, sr = sf.read(io.BytesIO(bytes(raw)), dtype="float32", always_2d=True)
    decoded = decoded.mean(axis=1)
    if sr != SR:
        import librosa
        decoded = librosa.resample(decoded, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return fit_length(decoded, audio.size, start=0) if decoded.size else audio


_CHANNEL_PROBE = None


def telephone(audio):
    """Reuse build_channel_probe.telephone: 300-3400 Hz, 8 kHz, mu-law, back to 16 kHz."""
    global _CHANNEL_PROBE
    if _CHANNEL_PROBE is None:
        spec = importlib.util.spec_from_file_location(
            "_ft_channel_probe", ROOT / "tools/build_channel_probe.py")
        _CHANNEL_PROBE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CHANNEL_PROBE)
    out, _details = _CHANNEL_PROBE.telephone(np.asarray(audio, dtype=np.float32))
    return out


def augment(audio, cfg, rng, music_reader=None):
    x = np.asarray(audio, dtype=np.float32)
    if cfg.p_music > 0 and cfg.music_files and rng.random() < cfg.p_music:
        music = music_reader(cfg.music_files[int(rng.integers(len(cfg.music_files)))], x.size, rng)
        x = mix_at_snr(x, music, rng.uniform(*cfg.music_snr_db))
    if cfg.p_noise > 0 and rng.random() < cfg.p_noise:
        x = add_noise(x, rng.uniform(*cfg.noise_snr_db), rng)
    if cfg.p_gain > 0 and rng.random() < cfg.p_gain:
        x = apply_gain(x, rng.uniform(*cfg.gain_db))
    channel = rng.random()
    try:
        if channel < cfg.p_mp3:
            x = mp3_roundtrip(x, cfg.mp3_kbps[int(rng.integers(len(cfg.mp3_kbps)))])
        elif channel < cfg.p_mp3 + cfg.p_telephone:
            x = telephone(x)
    except Exception as error:  # a failed channel sim must not kill training
        print(f"WARNING: channel augmentation failed: {error}", file=sys.stderr)
    if not np.isfinite(x).all():
        x = np.asarray(audio, dtype=np.float32)
    return x.astype(np.float32, copy=False)


def list_audio(folder):
    folder = Path(folder)
    return sorted(str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class CropDataset:
    """Map-style dataset: (normalized crop, target class, weight)."""

    def __init__(self, rows, seconds, train, aug=None, seed=0, stem_dir=None, trim_db=0.0):
        self.trim_db = trim_db
        self.rows = rows
        self.seconds = seconds
        self.train = train
        self.aug = aug or AugConfig()
        self.seed = seed
        self.stem_dir = stem_dir
        self._rng = None

    def __len__(self):
        return len(self.rows)

    def _stem(self, row):
        return stem_path(self.stem_dir, row["sha256"]) if self.stem_dir else None

    def _music(self, path, n, rng):
        return read_crop(path, n / SR, rng)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        if self.train:
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                self._rng = np.random.default_rng(info.seed if info else torch.initial_seed())
            rng = self._rng
        else:
            rng = None  # deterministic head crop, no augmentation
        audio = read_crop(row["path"], self.seconds, rng, stem_file=self._stem(row),
                          trim_db=self.trim_db)
        if self.train and self.aug.active():
            audio = augment(audio, self.aug, rng, self._music)
        target = FAKE_INDEX if row["label"] == 1 else 1 - FAKE_INDEX
        return (torch.from_numpy(normalize(audio)), torch.tensor(target, dtype=torch.long),
                torch.tensor(row["weight"], dtype=torch.float32))


def attach_hashes(rows, stem_dir):
    """Ensure every row has its source sha256 and a cached stem when stems are used."""
    missing = []
    for row in rows:
        row["sha256"] = sha256_file(row["path"])  # cache key = actual bytes, never trusted
        if stem_dir and not stem_path(stem_dir, row["sha256"]).is_file():
            missing.append(row["path"])
    return missing


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_frontend(path=FRONTEND_PATH, expected_sha=FRONTEND_SHA256, name="_ft_antideepfake_frontend"):
    if expected_sha and sha256_file(path) != expected_sha:
        raise ValueError(f"frontend SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_keys(native, weight_path):
    """Accept base (with pretraining tensors) and fine-tuned (inference-only) files."""
    report = native.audit_checkpoint_keys(native.read_safetensors_header(weight_path))
    if report["missing"] or report["unexpected"] or report["shape_mismatches"]:
        raise RuntimeError(native.format_audit_error(report, weight_path))
    return report


def build_model(native, weight_path=None):
    import torch
    from safetensors.torch import load_file
    model = native.build_native_detector_class(torch)()
    if weight_path is not None:
        check_keys(native, weight_path)
        state = load_file(str(weight_path), device="cpu")
        for key in native.PRETRAINING_ONLY_KEYS:
            state.pop(key, None)
        model.load_state_dict(state, strict=True)
    return model


def enable_grad_checkpointing(model):
    """Checkpoint each transformer layer; numerics are unchanged."""
    import torch
    from torch.utils.checkpoint import checkpoint
    for layer in model.m_ssl.model.encoder.layers:
        inner = layer.forward

        def forward(x, self_attn_padding_mask=None, _inner=inner):
            if torch.is_grad_enabled() and any(p.requires_grad for p in _inner.__self__.parameters()):
                return checkpoint(_inner, x, self_attn_padding_mask, use_reentrant=False)
            return _inner(x, self_attn_padding_mask)

        layer.forward = forward


def freeze(model, freeze_cnn=True, freeze_layers=0):
    ssl = model.m_ssl.model
    if freeze_cnn:
        ssl.feature_extractor.requires_grad_(False)
    if freeze_layers > 0:
        for module in (ssl.layer_norm, ssl.post_extract_proj, ssl.encoder.pos_conv):
            module.requires_grad_(False)
        for layer in ssl.encoder.layers[:freeze_layers]:
            layer.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable)


def param_groups(model, lr, lr_head, weight_decay):
    groups = {k: [] for k in ("head", "decay", "no_decay")}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("proj_fc."):
            groups["head"].append(param)
        elif param.ndim >= 2 and not name.endswith("weight_g"):
            groups["decay"].append(param)
        else:
            groups["no_decay"].append(param)
    out = [dict(params=groups["head"], lr=lr_head, weight_decay=weight_decay, name="head"),
           dict(params=groups["decay"], lr=lr, weight_decay=weight_decay, name="decay"),
           dict(params=groups["no_decay"], lr=lr, weight_decay=0.0, name="no_decay")]
    return [g for g in out if g["params"]]


def lr_factor(step, warmup, total, floor=0.1):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def inference_state(model, native):
    keys = native.expected_inference_keys()
    return {k: v.detach().to("cpu").float().contiguous() for k, v in model.state_dict().items()
            if k in keys}


def predict_fake(model, audio, device, max_samples=NII_MAX_AUDIO_SAMPLES, autocast=False):
    """Replicates script_sub55.predict_nii_fake (first 30 s, fp32, softmax[0,0])."""
    import torch
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < NII_MIN_AUDIO_SAMPLES or not np.isfinite(waveform).all():
        raise ValueError("NII received invalid or too-short voice audio")
    waveform = normalize(waveform[:max_samples])
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(tensor)
        probability = torch.softmax(logits.float(), dim=-1)[0, FAKE_INDEX]
    score = float(probability)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Unexpected NII score: {score}")
    return score


def binary_metrics(labels, scores):
    from sklearn.metrics import roc_auc_score
    from deepvoicehackathon.metrics import official_eer
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if np.unique(labels).size < 2:
        return dict(n=int(labels.size), eer=None, auc=None)
    return dict(n=int(labels.size), n_fake=int(labels.sum()),
                eer=official_eer(labels, scores), auc=float(roc_auc_score(labels, scores)))


def set_vram_cap(torch, device, cap_gb):
    if torch.device(device).type != "cuda" or not cap_gb:
        return None
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, cap_gb * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction


def free_commit_gb():
    """Free system commit in GB (= Win32_OperatingSystem.FreeVirtualMemory); RAM elsewhere."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPageFile / 1024 ** 3
    import psutil
    return psutil.virtual_memory().available / 1024 ** 3


def free_resources():
    """(free commit GB, free VRAM GB or None) without creating a CUDA context."""
    ram = free_commit_gb()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        vram = int(out.split()[0]) / 1024
    except Exception:
        vram = None
    return ram, vram


def wait_for_resources(min_ram_gb, min_vram_gb, interval_s=120, max_wait_s=None, label=""):
    """Block (retrying every interval) until free RAM/VRAM meet the thresholds."""
    started = time.time()
    while True:
        ram, vram = free_resources()
        ok = ram >= min_ram_gb and (not min_vram_gb or (vram is not None and vram >= min_vram_gb))
        vram_text = "n/a" if vram is None else f"{vram:.1f}"
        if ok:
            print(f"[resources{label}] ok: free commit {ram:.1f} GB, free VRAM {vram_text} GB", flush=True)
            return ram, vram
        if max_wait_s is not None and time.time() - started > max_wait_s:
            raise TimeoutError(f"resources not available after {max_wait_s}s")
        print(f"[resources{label}] waiting: free commit {ram:.1f}/{min_ram_gb} GB, free VRAM "
              f"{vram_text}/{min_vram_gb} GB; retry in {interval_s}s", flush=True)
        time.sleep(interval_s)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def evaluate(model, loader, device, amp):
    import torch
    model.eval()
    labels, scores, losses = [], [], []
    with torch.inference_mode():
        for wav, target, _w in loader:
            wav, target = wav.to(device), target.to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(wav)
            logits = logits.float()
            losses.append(float(torch.nn.functional.cross_entropy(logits, target, reduction="sum")))
            scores += torch.softmax(logits, -1)[:, FAKE_INDEX].cpu().tolist()
            labels += (target == FAKE_INDEX).long().cpu().tolist()
    model.train()
    out = binary_metrics(labels, scores)
    out["loss"] = sum(losses) / max(1, len(labels))
    return out, scores


def slice_metrics(rows, scores):
    """Per-channel (fakes vs reals of that channel) and per-system (fakes vs all reals)."""
    labels = [r["label"] for r in rows]
    reals = [i for i, y in enumerate(labels) if y == 0]
    out = dict(channel={}, system={})
    for key, bucket in (("channel", "channel"), ("system", "system")):
        values = sorted({r["meta"].get(key, "") for r in rows})
        for value in values:
            idx = [i for i, r in enumerate(rows) if r["meta"].get(key, "") == value]
            if key == "system":
                if all(labels[i] == 0 for i in idx):
                    out[bucket][value] = dict(n=len(idx), mean_fake_prob=float(np.mean([scores[i] for i in idx])))
                    continue
                idx = sorted(set(idx) | set(reals))
            m = binary_metrics([labels[i] for i in idx], [scores[i] for i in idx])
            fake_idx = [i for i in idx if labels[i] == 1 and rows[i]["meta"].get(key, "") == value]
            m["mean_fake_prob"] = float(np.mean([scores[i] for i in fake_idx])) if fake_idx else None
            out[bucket][value] = m
    return out


def sampling_weights(rows, balance=True, mix=None):
    """Per-row draw weights: optional per-manifest shares, labels balanced within each."""
    sources = sorted({r.get("source_index", 0) for r in rows})
    if mix:
        if len(mix) != len(sources):
            raise ValueError(f"--mix needs {len(sources)} shares, got {len(mix)}")
        shares = {s: float(m) / float(sum(mix)) for s, m in zip(sources, mix)}
    else:
        shares = None
    cells = {}
    for r in rows:
        key = (r.get("source_index", 0) if shares else 0, r["label"] if balance else 0)
        cells[key] = cells.get(key, 0.0) + r["weight"]
    weights = []
    for r in rows:
        source = r.get("source_index", 0) if shares else 0
        key = (source, r["label"] if balance else 0)
        n_cells = sum(1 for k in cells if k[0] == source)
        share = shares[source] if shares else 1.0
        weights.append(share * r["weight"] / (cells[key] * n_cells))
    return weights


def train(args, native=None):
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, WeightedRandomSampler

    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=bool(args.overwrite))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if args.wait_ram_gb or args.wait_vram_gb:
        wait_for_resources(args.wait_ram_gb, args.wait_vram_gb if device != "cpu" else 0,
                           label=" train")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cap_fraction = set_vram_cap(torch, device, args.vram_cap_gb)

    # --- data
    manifests, rows = [], []
    for index, path in enumerate(args.manifest):
        part = subsample(read_manifest(path, tag=f"m{index}"), args.limit_per_manifest, args.seed)
        for row in part:
            row["source_index"] = index
        rows += part
        manifests.append(dict(path=str(Path(path).resolve()), sha256=sha256_file(path),
                              rows_used=len(part)))
    if args.val_manifest:
        train_rows = rows
        val_rows = []
        for index, path in enumerate(args.val_manifest):
            val_rows += read_manifest(path, tag=f"v{index}")
            manifests.append(dict(path=str(Path(path).resolve()), sha256=sha256_file(path),
                                  role="val"))
    else:
        train_rows, val_rows = group_split(rows, args.val_fraction, args.seed)
    stem_dir = Path(args.stems) if args.stems else None
    if stem_dir:
        missing = attach_hashes(train_rows + val_rows, stem_dir)
        if missing:
            raise FileNotFoundError(f"{len(missing)} rows lack cached stems in {stem_dir}; "
                                    f"first: {missing[:3]}")
    overlap = {r["group"] for r in train_rows} & {r["group"] for r in val_rows}
    if overlap and not args.val_manifest:
        raise AssertionError("train/val group overlap")
    music = []
    for folder in args.music_dir or []:
        music += list_audio(folder)
    for listing in args.music_list or []:
        with open(listing, encoding="utf-8-sig") as handle:
            music += [line.strip() for line in handle if line.strip() and line.strip() != "path"]
    aug = AugConfig(p_gain=args.p_gain, p_noise=args.p_noise, p_mp3=args.p_mp3,
                    p_telephone=args.p_telephone, p_music=args.p_music if music else 0.0,
                    music_files=music, music_snr_db=tuple(args.music_snr))
    train_set = CropDataset(train_rows, args.crop_seconds, True, aug, args.seed, stem_dir, args.trim_db)
    val_set = CropDataset(val_rows, args.val_seconds, False, None, args.seed, stem_dir, args.trim_db)
    weights = sampling_weights(train_rows, args.balance, args.mix)
    total_draws = max(1, args.max_steps) * args.batch_size * args.grad_accum
    sampler = WeightedRandomSampler(weights, num_samples=total_draws, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader_kw = dict(num_workers=args.workers, pin_memory=device == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, **loader_kw)

    # --- model
    native = native or load_frontend()
    base = BASES.get(args.base, {})
    base_path = Path(args.base_path) if args.base_path else base["path"]
    base_sha = sha256_file(base_path)
    if not args.base_path and base_sha != base["sha256"]:
        raise ValueError(f"base checkpoint SHA mismatch for {args.base}: {base_sha}")
    model = build_model(native, base_path)
    params = freeze(model, not args.train_cnn, args.freeze_layers)
    if args.grad_checkpointing:
        enable_grad_checkpointing(model)
    model.to(device).train()
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.lr_head, args.weight_decay),
                                  betas=(0.9, 0.98), eps=1e-8)
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    amp = bool(args.bf16) and device == "cuda"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"train={len(train_rows)} val={len(val_rows)} groups={len({r['group'] for r in train_rows})}"
          f"/{len({r['group'] for r in val_rows})} params trainable={params['trainable']:,}"
          f"/{params['total']:,} cap_fraction={cap_fraction}", flush=True)

    history, best, bad_rounds, step = [], None, 0, 0
    ckpt_path = out_dir / "model.safetensors"

    def save_best(metrics):
        state = inference_state(model, native)
        save_file(state, str(ckpt_path), metadata=dict(
            base=args.base, base_sha256=base_sha, step=str(step),
            frontend_sha256=FRONTEND_SHA256, format="nii-antideepfake-native-v1"))

    def run_eval(tag):
        nonlocal best, bad_rounds
        if val_rows:
            metrics, val_scores = evaluate(model, val_loader, device, amp)
            metrics["slices"] = slice_metrics(val_rows, val_scores)
        else:
            metrics = dict(eer=None)
        metrics.update(step=step, tag=tag, elapsed_s=round(time.time() - started, 1))
        history.append(metrics)
        key = (metrics["eer"] if metrics["eer"] is not None else math.inf,
               metrics.get("loss", math.inf))
        improved = best is None or key < best[0]
        if improved:
            best = (key, dict(metrics))
            bad_rounds = 0
            save_best(metrics)
        else:
            bad_rounds += 1
        print(json.dumps(dict(eval=metrics, improved=improved)), flush=True)
        return improved

    run_eval("step0")
    train_started = time.time()
    step_times, losses, data_iter = [], [], iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        t0 = time.time()
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            wav, target, weight = next(data_iter)
            wav, target, weight = (wav.to(device, non_blocking=True), target.to(device),
                                   weight.to(device))
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(wav)
            loss = torch.nn.functional.cross_entropy(
                logits.float(), target, reduction="none", label_smoothing=args.label_smoothing)
            loss = (loss * weight).sum() / weight.sum().clamp_min(1e-8) / args.grad_accum
            loss.backward()
            accum_loss += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       args.clip)
        factor = lr_factor(step, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        step += 1
        step_times.append(time.time() - t0)
        losses.append(accum_loss)
        if step % args.log_every == 0 or step == args.max_steps:
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device == "cuda" else 0.0
            print(f"step {step} loss {np.mean(losses[-args.log_every:]):.4f} "
                  f"s/step {np.mean(step_times[-args.log_every:]):.3f} peak_alloc_gb {peak:.2f}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            run_eval("periodic")
            if args.patience and bad_rounds >= args.patience:
                print(f"early stop at step {step}", flush=True)
                break
        if args.max_minutes and time.time() - train_started > 60 * args.max_minutes:
            print(f"time cap reached at step {step}", flush=True)
            if step % args.eval_every:
                run_eval("time_cap")
            break

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device == "cuda" else None
    steady = step_times[1:] if len(step_times) > 1 else step_times
    import transformers  # noqa: F401  (version provenance only)
    receipt = dict(
        tool="tools/ft_antideepfake.py", git_head=git_head(), created_unix=int(started),
        base=args.base, base_path=str(base_path), base_sha256=base_sha,
        frontend_path=str(FRONTEND_PATH), frontend_sha256=FRONTEND_SHA256,
        manifests=manifests, seed=args.seed, config=vars(args),
        split=dict(train=len(train_rows), val=len(val_rows),
                   train_fake=int(sum(r["label"] for r in train_rows)),
                   val_fake=int(sum(r["label"] for r in val_rows)),
                   train_groups=len({r["group"] for r in train_rows}),
                   val_groups=len({r["group"] for r in val_rows})),
        val_paths=[r["path"] for r in val_rows] if len(val_rows) <= 5000 else None,
        params=params, steps_run=step, best=best[1] if best else None, history=history,
        train_loss_tail=float(np.mean(losses[-args.log_every:])) if losses else None,
        seconds_per_step=float(np.mean(steady)) if steady else None,
        peak_vram_alloc_gb=peak_alloc, peak_vram_reserved_gb=peak_reserved,
        vram_cap_gb=args.vram_cap_gb, versions=dict(torch=torch.__version__,
                                                    transformers=transformers.__version__),
        checkpoint=str(ckpt_path), checkpoint_sha256=sha256_file(ckpt_path),
        output_index_fake=FAKE_INDEX, wall_s=round(time.time() - started, 1))
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps(dict(best=receipt["best"], peak_vram_alloc_gb=peak_alloc,
                          peak_vram_reserved_gb=peak_reserved,
                          seconds_per_step=receipt["seconds_per_step"])), flush=True)
    return receipt


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--val-manifest", action="append")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--base", choices=sorted(BASES), default="wav2vec-large")
    p.add_argument("--base-path", help="override base weights (skips pinned SHA check)")
    p.add_argument("--stems", nargs="?", const=str(DEFAULT_STEM_DIR),
                   help="train/val from cached vocal stems DIR/<sha256>.npy")
    p.add_argument("--limit-per-manifest", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--crop-seconds", type=float, default=4.0)
    p.add_argument("--val-seconds", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock cap for the step loop")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=5, help="eval rounds without EER gain; 0=off")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-balance", dest="balance", action="store_false")
    p.add_argument("--mix", type=float, nargs="+",
                   help="per --manifest sampling shares (labels balanced within each)")
    p.add_argument("--train-cnn", action="store_true", help="unfreeze the CNN feature encoder")
    p.add_argument("--freeze-layers", type=int, default=8)
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--p-gain", type=float, default=0.3)
    p.add_argument("--p-noise", type=float, default=0.3)
    p.add_argument("--p-mp3", type=float, default=0.2)
    p.add_argument("--p-telephone", type=float, default=0.1)
    p.add_argument("--p-music", type=float, default=0.0)
    p.add_argument("--music-dir", action="append")
    p.add_argument("--music-list", action="append", help="text file: one music path per line")
    p.add_argument("--music-snr", type=float, nargs=2, default=(0.0, 20.0))
    p.add_argument("--trim-db", type=float, default=0.0,
                   help="trim leading/trailing silence (top_db) before cropping; 0=off")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vram-cap-gb", type=float, default=8.0)
    p.add_argument("--wait-ram-gb", type=float, default=0.0, help="wait until this much system commit (GB) is free")
    p.add_argument("--wait-vram-gb", type=float, default=0.0, help="wait until this much VRAM is free")
    p.add_argument("--seed", type=int, default=20260924)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.p_mp3 + args.p_telephone > 1:
        raise SystemExit("--p-mp3 + --p-telephone must be <= 1")
    train(args)


if __name__ == "__main__":
    main()
