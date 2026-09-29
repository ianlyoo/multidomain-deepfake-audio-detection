"""sub102 CPU tests: runner transform, window/pool parity with the eval code, fallbacks, window labels.

  .venv/Scripts/python.exe -B tests/test_sub102_nii_ml.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/sub102r3_config.json"


def _config_bytes():
    if CONFIG.exists():
        return CONFIG.read_bytes()
    return json.dumps(dict(source_runner=str(ROOT / "submission/script_sub100.py"), weights_sha256="0" * 64,
                           pool=dict(file=dict(tau=1.0, gamma=0.0, a=1.0, b=0.0),
                                     voice=dict(tau=0.5, gamma=1.0, a=0.9, b=-0.1),
                                     music=dict(tau=2.0, gamma=2.0, a=1.1, b=0.2)),
                           weights=dict(file=0.3, voice=0.3, music=0.3))).encode()


def _runner():
    import make_sub102_runner as m
    cfg = json.loads(_config_bytes())
    source = Path(cfg.get("source_runner", ROOT / "submission/script_sub100.py"))
    src = source.read_text(encoding="utf-8")
    text = m.build(src, _config_bytes())
    tmp = Path(tempfile.mkdtemp()) / "script_sub102_test.py"
    tmp.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("sub102_test_runner", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return src, text, mod, cfg


class Toy:
    """Deterministic stand-in for the network: per-window statistics -> 5 logits."""

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, wav):
        t = self.torch
        x = wav.float()
        return t.stack([x.mean(1) * 3, x[:, :100].mean(1), x.abs().mean(1) - 1, x[:, -50:].mean(1), x.std(1) - 1], 1)


def test_transform_additive_and_pins():
    src, text, mod, cfg = _runner()
    remaining = iter(text.splitlines())
    assert all(any(line == new for new in remaining) for line in src.splitlines())
    for name in ("load_sub102_model", "_sub102_window_logits", "_sub102_pool", "_sub102_file_probs"):
        assert hasattr(mod, name)
    assert mod.SUB102_SHA256 == cfg["weights_sha256"]
    assert text.count("sub102_probs = _sub102_file_probs(") == 1
    # FILE blend sits before the floor, VOICE/MUSIC blends after the sub90 publish
    assert text.index("SUB102_FILE_WEIGHT), 10)") < text.index('row["FILE_FAKE_PROB"] = _sub52_light_file(row)')
    assert text.index('row["VOICE_FAKE_PROB"] = round(sub90_output_voice, 10)') < text.index("SUB102_VOICE_WEIGHT), 10)")


def test_window_logits_match_scorer():
    import torch
    import ft_antideepfake as ft
    import ft_nii_multilabel as ml
    _, _, mod, _ = _runner()
    toy = Toy(torch)
    rng = np.random.default_rng(0)
    for n in (5000, 64000, 64001, 150000, 16000 * 400):  # short, exact, +1, typical, > cap (400 s)
        audio = (rng.standard_normal(n) * 0.1).astype(np.float32)
        z1 = mod._sub102_window_logits(toy, audio, "cpu")
        z2, starts = ml.score_windows(toy, audio, "cpu")
        assert z1.shape == z2.shape, (n, z1.shape, z2.shape)
        assert np.max(np.abs(z1 - z2)) == 0.0
        assert len(starts) <= ml.MAX_WINDOWS
    _ = ft


def test_pool_matches_eval():
    import sub102_eval as ev
    _, _, mod, _ = _runner()
    rng = np.random.default_rng(1)
    for _ in range(50):
        L = rng.normal(0, 3, size=(int(rng.integers(1, 30)), 5))
        rec = dict(logits=L.tolist())
        for head, h, ph in (("file", 0, None), ("voice", 1, 3), ("music", 2, 4)):
            for tau in (0.25, 1.0, 1e3):
                for gamma in (0.0, 1.0, 2.0):
                    pool = dict(tau=tau, gamma=gamma, a=0.8, b=-0.3)
                    want = ev.ft_prob(rec, head, pool)
                    got = mod._sub102_pool(L, h, ph, tau, gamma, 0.8, -0.3)
                    assert abs(want - got) < 1e-12, (head, tau, gamma, want, got)


def test_fallbacks():
    import torch
    _, _, mod, _ = _runner()
    audio = np.zeros(32000, np.float32) + 0.01
    assert mod._sub102_file_probs(None, audio, "cpu") is None
    assert mod._sub102_file_probs(Toy(torch), None, "cpu") is None

    def broken(_wav):
        raise RuntimeError("boom")
    assert mod._sub102_file_probs(broken, audio, "cpu") is None
    bad = audio.copy(); bad[5] = np.nan
    assert mod._sub102_file_probs(Toy(torch), bad, "cpu") is None
    p = mod._sub102_file_probs(Toy(torch), audio, "cpu")
    assert p is not None and len(p) == 3 and all(0 < x < 1 for x in p)
    try:
        mod.load_sub102_model(None, "cpu")
        raise AssertionError("load without base must fail")
    except RuntimeError:
        pass


def test_window_labels():
    import ft_nii_multilabel as ml
    sr = ml.SR
    item = dict(source="mixture_v2", kind="partial", comps=[[0, 3 * sr, "speech", 0], [3 * sr, 5 * sr, "speech", 1],
                                                             [5 * sr, 12 * sr, "speech", 0]])
    y, m = ml.window_labels(item, 0)            # 1 s of splice inside -> fake
    assert (y[0], m[0], y[1], m[1]) == (1, 1, 1, 1) and m[2] == 0 and (y[3], m[3]) == (1, 1)
    y, m = ml.window_labels(item, 8 * sr)       # no splice -> real
    assert (y[0], m[0], y[1], m[1]) == (0, 1, 0, 1)
    y, m = ml.window_labels(item, int(4.8 * sr))  # 0.2 s splice -> ambiguous, masked
    assert m[0] == 0 and m[1] == 0
    seq = dict(source="mixture_v2", kind="seq_rv", comps=[[0, 6 * sr, "speech", 0], [6 * sr, 12 * sr, "music", 1]])
    y, m = ml.window_labels(seq, 0)             # speech only, real
    assert (y[0], m[0], y[1], m[1], y[4], m[4]) == (0, 1, 0, 1, 0, 1) and m[2] == 0
    y, m = ml.window_labels(seq, 8 * sr)        # fake music only
    assert (y[0], m[0], y[2], m[2]) == (1, 1, 1, 1) and m[1] == 0
    view = dict(source="sub99_view", kind="view", file_fake=None, mask_music=True, comps=[[0, 5 * sr, "speech", 0]])
    y, m = ml.window_labels(view, 0)
    assert m[0] == 0 and (y[1], m[1]) == (0, 1) and m[2] == 0 and m[4] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)} passed")
