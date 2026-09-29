"""Local-only, inference-only Nes2Net-X singing checkpoint adapter.

The source bundle is explicit and hash-checked; no hub, download or fallback.
See docs/nes2net_forward_audit.md for provenance and unvalidated use cases.
Raw scores point toward real audio; their negatives are fake ranking scores,
not probabilities. Variable-length batches are intentionally unsupported.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import types

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_ROOT = Path(os.environ.get('DATA_DIR', 'data')) / 'models/nes2net-singing'
CHECKPOINT_NAME = 'WavLM_Nes2Net_X_e75_seed420_valid0.03192785031473534.pt'
CHECKPOINT_SIZE = 1264329467
CHECKPOINT_SHA256 = 'f1c2942635c68b584a742b10dbc5630759fc6cd55409ff8dc4d71f9c81ccdf5f'
S3PRL_REVISION = '53193b5b2fe7b851cc43236c2fc5f2dc37ac3378'
AUTHOR_REVISION = '3fed08a603991cf8b422978e0c942a976deb9399'
# Digests of the stored UTF-8/LF source bundle (one terminal blank line added).
SOURCE_HASHES = {
    'modules.py': '30e866916955976cba827f9ec9882531e0f6cec776ed29db818e7584d197a4f0',
    'WavLM.py': '4b5ea55668cfdc2300261f75003ade06e72ba22082f096fe05d331d727b6e99f',
    'WavLM_Nes2Net_X.py': '99f16ab058745d546bbf1ac3e4bd5cd44f80f48cbb6c76c429e6924174766219',
    'expert.py': '1840113e47597cbfad727fd29d048fbac837dfb31616dcc6b8276f0226ef5e27',
    'interfaces.py': '8ea45c81e544f3a6ed4c59c512d683432a0206f8722a5f0cc4ba10a57831d702',
}
CONFIG = dict(
    extractor_mode='layer_norm', encoder_layers=24, encoder_embed_dim=1024,
    encoder_ffn_embed_dim=4096, encoder_attention_heads=16, activation_fn='gelu',
    layer_norm_first=True, conv_bias=False, normalize=True,
    relative_position_embedding=True, num_buckets=320, max_distance=800,
    gru_rel_pos=True, encoder_layerdrop=0.0, feature_grad_mult=0.0,
)


def file_sha256(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_source(root: Path, name: str) -> str:
    data = (Path(root) / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError(f'Source digest mismatch: {name}')
    return data.decode('utf-8')


def source_classes(text: str, names: set[str], namespace: dict) -> dict:
    """Execute only requested class definitions, never imports/hub/main blocks."""
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Pinned source class selection is incomplete')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-nes2net>', 'exec'), namespace)
    return namespace


def load_source_bundle(root: Path):
    # Verify on every construction, including when Python modules already exist.
    texts = {n: checked_source(root, n) for n in SOURCE_HASHES}
    package_name = '_deepvoice_nes2net_pinned_' + S3PRL_REVISION
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        try:
            for name in ('modules', 'WavLM'):
                module = types.ModuleType(package_name + '.' + name)
                module.__package__ = package_name
                sys.modules[module.__name__] = module
                exec(compile(texts[name + '.py'], str(Path(root) / (name + '.py')), 'exec'), module.__dict__)
        except Exception:
            for name in (package_name, package_name + '.modules', package_name + '.WavLM'):
                sys.modules.pop(name, None)
            raise
    namespace = source_classes(texts['WavLM_Nes2Net_X.py'], {
        'ASTP', 'SEModule', 'Bottle2neck', 'Nested_Res2Net_TDNN',
        'SSLModel', 'WavLM_Nes2Net_noRes_w_allT',
    }, dict(torch=torch, nn=nn, F=F, math=math))
    return sys.modules[package_name + '.WavLM'], namespace


def prepare_waveform(waveform, *, sample_rate: int = 16000, mode: str = 'full'):
    """Mono float32 16 kHz. Crop/repeat BEFORE S3PRL layer normalization."""
    if sample_rate != 16000:
        raise ValueError('Tensor input must already be resampled to 16000 Hz')
    x = torch.as_tensor(waveform, dtype=torch.float32)
    if x.ndim != 1 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError('Expected a finite, nonempty mono waveform')
    if mode == '4s':
        x = x.repeat(64000 // x.numel() + 1)[:64000] if x.numel() < 64000 else x[:64000]
    elif mode != 'full':
        raise ValueError("mode must be 'full' or '4s'")
    # Seven valid convolutions have a 400-sample receptive field.
    if x.numel() < 400:
        raise ValueError('Full-mode input needs at least 400 samples; choose 4s explicitly to tile')
    return x


def normalize_batch(waves: torch.Tensor):
    if waves.ndim != 2 or waves.shape[0] == 0 or waves.shape[1] < 400:
        raise ValueError('Expected equal-length [batch, samples>=400] tensor')
    if waves.dtype != torch.float32 or not torch.isfinite(waves).all():
        raise ValueError('Expected finite float32 waveforms')
    # Keep the source operation order; per-row norm precedes batch formation.
    normalized = torch.stack([F.layer_norm(w, w.shape) for w in waves])
    return normalized, torch.zeros_like(normalized, dtype=torch.bool)


def collect_hidden_states(frontend, waves: torch.Tensor):
    normalized, padding_mask = normalize_batch(waves)
    states = []
    handles = [layer.register_forward_hook(
        lambda module, inputs, output: states.append(inputs[0].transpose(0, 1)))
        for layer in frontend.encoder.layers]
    handles.append(frontend.encoder.register_forward_hook(
        lambda module, inputs, output: states.append(output[0])))
    try:
        frontend.extract_features(normalized, padding_mask=padding_mask, mask=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(states) != len(frontend.encoder.layers) + 1:
        raise RuntimeError('Incomplete hidden-state capture')
    return tuple(states)


def sea_merge(states, gate):
    stacked = torch.stack(states, dim=1)
    b, c, _, _ = stacked.shape
    weights = gate(F.adaptive_avg_pool2d(stacked, 1).view(b, c)).view(b, c, 1, 1)
    return (stacked * weights.expand_as(stacked)).sum(dim=1)


class Nes2NetSinging(nn.Module):
    """Float32/eval-only model; serializes forwards to isolate transient hooks.

    Construct through from_checkpoint. ``forward`` takes [B,T] pre-resampled
    equal-length raw audio, returns [B,1] real-oriented scores. ``score`` applies
    explicit full/4s preparation to a single mono input. No implicit clipping,
    source separation, calibration, fallback or padded mixed-length batching.
    """

    def __init__(self, frontend, backend):
        super().__init__()
        self.ssl_model = nn.Module()
        self.ssl_model.model = nn.Module()
        self.ssl_model.model.model = frontend
        self.ssl_model.fc_att_merge = nn.Sequential(
            nn.Linear(25, 8, bias=False), nn.ReLU(inplace=True),
            nn.Linear(8, 25, bias=False), nn.Sigmoid())
        self.Nested_Res2Net_TDNN = backend
        self._forward_lock = threading.Lock()
        self.load_report = None

    @classmethod
    def from_checkpoint(cls, checkpoint=DEFAULT_ROOT / CHECKPOINT_NAME,
                        source_dir=DEFAULT_ROOT / 'source', device='cpu'):
        checkpoint = Path(checkpoint)
        if checkpoint.stat().st_size != CHECKPOINT_SIZE:
            raise ValueError('Incomplete or wrong checkpoint size')
        if file_sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint SHA256 mismatch')
        sd = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if not isinstance(sd, dict) or len(sd) != 1050 or not all(torch.is_tensor(v) for v in sd.values()):
            raise ValueError('Expected exactly 1050 checkpoint tensors')
        wavlm, author = load_source_bundle(Path(source_dir))
        with torch.device('cpu'):
            cfg = wavlm.WavLMConfig(CONFIG)
            model = cls(wavlm.WavLM(cfg), author['Nested_Res2Net_TDNN'](
                Nes_ratio=[8, 8], input_channel=1024, dilation=1,
                pool_func='mean', SE_ratio=[1]))
        expected = model.state_dict()
        if len(expected) != 1050 or set(expected) != set(sd):
            raise ValueError('Checkpoint key mismatch; no initialized-tensor fallback allowed')
        if any(expected[k].shape != sd[k].shape or expected[k].dtype != sd[k].dtype for k in expected):
            raise ValueError('Checkpoint shape/dtype mismatch')
        result = model.load_state_dict(sd, strict=True)
        if not all(torch.equal(v, sd[k]) for k, v in model.state_dict().items()):
            raise RuntimeError('Loaded tensor equality check failed')
        model.load_report = dict(entries=len(sd), missing=len(result.missing_keys),
                                 unexpected=len(result.unexpected_keys), all_tensors_equal=True)
        model.config = cfg
        model.source_dir = Path(source_dir)
        return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @property
    def frontend(self):
        return self.ssl_model.model.model

    @torch.inference_mode()
    def forward(self, waves):
        if any(module.training for module in self.modules()):
            raise RuntimeError('This adapter is inference-only; call eval()')
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError('Only float32 inference has been audited')
        waves = waves.to(device=parameter.device)
        with self._forward_lock:
            states = collect_hidden_states(self.frontend, waves)
            merged = sea_merge(states, self.ssl_model.fc_att_merge)
            scores = self.Nested_Res2Net_TDNN(merged.permute(0, 2, 1))
        if not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite Nes2Net score; no fallback')
        return scores

    def score(self, waveform, *, sample_rate=16000, mode='full') -> float:
        x = prepare_waveform(waveform, sample_rate=sample_rate, mode=mode)
        return self(x.unsqueeze(0)).item()

    def score_file(self, path, *, mode='full') -> float:
        import librosa
        wave, sr = librosa.load(path, sr=16000, mono=True)
        return self.score(wave, sample_rate=sr, mode=mode)
