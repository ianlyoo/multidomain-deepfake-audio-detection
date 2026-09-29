"""sub100 (or later) runner -> sub102 runner: NII multi-label fine-tune on raw-mixture windows.

A copy of the already-loaded NII AntiDeepfake model is overlaid with the fine-tuned tensors
(model/nii_ml_sub102.safetensors, fp16-stored; the frozen CNN and layers 0-7 are the base
weights) and a 5-way linear head (FILE/VOICE/MUSIC-fake, VOICE/MUSIC-present). Each file's
16 kHz mixture (the runner's own load_audio) is cut into 4 s windows with a 2 s hop, each
window normalized like predict_nii_fake and scored under bf16 autocast in batches of 16.
Per head, a learned smooth-max pool (temperature tau, presence-weight exponent gamma,
affine a/b; fit on the mixture_v2 val split) gives a probability that is logit-blended into:
  FILE  - the pre-floor FILE (rounded row value) before the unchanged floor,
  VOICE - the published VOICE after the floor (the floor still reads sub70 VOICE),
  MUSIC - the published MUSIC after the floor (the floor still reads whole-file MUSIC).
Any failure (model load, missing mixture audio, window scoring) keeps the exact source outputs.
Additive only.

  .venv/Scripts/python.exe tools/make_sub102_runner.py --source submission/script_sub100.py \
      --config $DATA_DIR/derived/mixture_v2/sub102_config.json --out submission/script_sub102.py
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

ANCHOR_HELPERS = "def predict_fake_scores_for_all_files(\n"
ANCHOR_LOAD = '    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):\n'
ANCHOR_INIT = "        sub90_output_voice = None\n"
ANCHOR_FILE = '        row["FILE_FAKE_PROB"] = _sub52_light_file(row)\n'
ANCHOR_PUBLISH = '            row["VOICE_FAKE_PROB"] = round(sub90_output_voice, 10)\n'
ANCHOR_SCORE = "            if sub94_file is not None:\n                file_fake = sub94_file\n"

LOAD = '''    # sub102: the new optional model loads only after every source model attempt.
    try:
        sub102_model = load_sub102_model(nii_model, device)
    except (Exception, SystemExit) as sub102_load_error:
        print(f"WARNING: sub102 NII multi-label model failed to load: {sub102_load_error}; "
              "retaining source outputs", file=sys.stderr)
        sub102_model = None

'''
INIT = "        sub102_probs = None\n"
SCORE = '''            # sub102: FT raw-mixture window heads; None keeps every source output.
            sub102_probs = _sub102_file_probs(sub102_model, mixture_audio, device)
'''
FILE = '''        # sub102: pre-floor FILE blend; the unchanged floor below reads it.
        if sub102_probs is not None:
            row["FILE_FAKE_PROB"] = round(_sub68_logit_blend(row["FILE_FAKE_PROB"], sub102_probs[0],
                                                              SUB102_FILE_WEIGHT), 10)
'''
PUBLISH = '''        # sub102: published VOICE/MUSIC blends after the floor has consumed the source values.
        if sub102_probs is not None:
            if SUB102_VOICE_WEIGHT:
                row["VOICE_FAKE_PROB"] = round(_sub68_logit_blend(row["VOICE_FAKE_PROB"], sub102_probs[1],
                                                                   SUB102_VOICE_WEIGHT), 10)
            if SUB102_MUSIC_WEIGHT:
                row["MUSIC_FAKE_PROB"] = round(_sub68_logit_blend(row["MUSIC_FAKE_PROB"], sub102_probs[2],
                                                                   SUB102_MUSIC_WEIGHT), 10)
'''


def helpers(cfg):
    pools = tuple((float(p["tau"]), float(p["gamma"]), float(p["a"]), float(p["b"]))
                  for p in (cfg["pool"]["file"], cfg["pool"]["voice"], cfg["pool"]["music"]))
    return '''# BEGIN SUB102 NII MULTI-LABEL HELPERS
# sub102: anchored multi-label fine-tune of the NII backbone (config sha256 %s).
SUB102_PATH = MODEL_DIR / "nii_ml_sub102.safetensors"
SUB102_SHA256 = "%s"
SUB102_WINDOW = 4 * AUDIO_SAMPLE_RATE
SUB102_HOP = 2 * AUDIO_SAMPLE_RATE
SUB102_BATCH = 16
SUB102_MAX_WINDOWS = 150
# (tau, gamma, a, b) per FILE/VOICE/MUSIC head; presence heads are 3 (VOICE) and 4 (MUSIC).
SUB102_POOL = %r
SUB102_FILE_WEIGHT = %r
SUB102_VOICE_WEIGHT = %r
SUB102_MUSIC_WEIGHT = %r


def load_sub102_model(nii_model, device):
    """Copy the loaded NII model, overlay the fine-tuned tensors, add the 5-way head."""
    import copy
    import hashlib
    from safetensors.torch import load_file
    if nii_model is None:
        raise RuntimeError("base NII model unavailable")
    digest = hashlib.sha256()
    with SUB102_PATH.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SUB102_SHA256:
        raise RuntimeError("sub102 weights SHA256 mismatch")
    overlay = load_file(str(SUB102_PATH), device="cpu")
    head_w = overlay.pop("heads.weight").float()
    head_b = overlay.pop("heads.bias").float()
    det = copy.deepcopy(nii_model).to("cpu")
    own = det.state_dict()
    state = {}
    for key, value in overlay.items():
        if not key.startswith("det.") or key[4:] not in own or own[key[4:]].shape != value.shape:
            raise RuntimeError("unexpected sub102 tensor: " + key)
        state[key[4:]] = value.float()
    det.load_state_dict(state, strict=False)
    if head_w.shape != (5, own["proj_fc.weight"].shape[1]) or head_b.shape != (5,):
        raise RuntimeError("sub102 head shape mismatch")

    class Sub102Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.det = det
            self.heads = torch.nn.Linear(head_w.shape[1], 5)
            with torch.no_grad():
                self.heads.weight.copy_(head_w)
                self.heads.bias.copy_(head_b)

        def forward(self, wav):
            emb = self.det.m_ssl.extract_feat(wav)
            return self.heads(emb.mean(dim=1))

    return Sub102Net().to(device).eval()


def _sub102_window_logits(model, audio, device):
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < NII_MIN_AUDIO_SAMPLES or not np.isfinite(waveform).all():
        raise ValueError("sub102 received invalid or too-short audio")
    starts = _sub80_window_starts(waveform.size, SUB102_WINDOW, SUB102_HOP)
    if len(starts) > SUB102_MAX_WINDOWS:
        keep = np.linspace(0, len(starts) - 1, SUB102_MAX_WINDOWS).round().astype(int).tolist()
        starts = [starts[i] for i in sorted(set(keep))]
    crops = []
    for start in starts:
        values = waveform[start : start + SUB102_WINDOW].astype(np.float64, copy=False)
        crops.append(((values - values.mean()) / np.sqrt(values.var() + 1e-5)).astype(np.float32))
    device = torch.device(device) if not isinstance(device, torch.device) else device
    out = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        for i in range(0, len(crops), SUB102_BATCH):
            chunk = crops[i : i + SUB102_BATCH]
            if len({c.size for c in chunk}) == 1:
                out.append(model(torch.from_numpy(np.stack(chunk)).to(device)).float().cpu().numpy())
            else:
                for c in chunk:
                    out.append(model(torch.from_numpy(c)[None].to(device)).float().cpu().numpy())
    logits = np.concatenate(out, 0).astype(np.float64)
    if logits.ndim != 2 or logits.shape[1] != 5 or not np.isfinite(logits).all():
        raise ValueError("Invalid sub102 window logits")
    return logits


def _sub102_pool(logits, head, presence_head, tau, gamma, a, b):
    z = logits[:, head]
    if presence_head is None or gamma == 0:
        w = np.ones_like(z)
    else:
        w = (1.0 / (1.0 + np.exp(-logits[:, presence_head]))) ** gamma + 1e-6
    w = w / max(w.sum(), 1e-12)
    zmax = z.max()
    pooled = float(tau * np.log(np.sum(w * np.exp((z - zmax) / tau))) + zmax)
    x = a * pooled + b
    return float(1.0 / (1.0 + np.exp(-x))) if x >= 0 else float(np.exp(x) / (1.0 + np.exp(x)))


def _sub102_file_probs(model, mixture_audio, device):
    """(FILE, VOICE, MUSIC) FT probabilities, or None to keep the source outputs."""
    if model is None or mixture_audio is None:
        return None
    try:
        logits = _sub102_window_logits(model, mixture_audio, device)
        probs = tuple(_sub102_pool(logits, h, ph, *SUB102_POOL[h])
                      for h, ph in ((0, None), (1, 3), (2, 4)))
        if not all(_sub37_valid_probability(p) for p in probs):
            raise ValueError("Invalid sub102 probability")
        return probs
    except (Exception, SystemExit) as error:
        print(f"WARNING: sub102 window scoring failed: {error}; retaining source outputs",
              file=sys.stderr)
        return None
# END SUB102 NII MULTI-LABEL HELPERS


''' % (cfg["_sha256"], cfg["weights_sha256"], pools, float(cfg["weights"]["file"]),
       float(cfg["weights"]["voice"]), float(cfg["weights"]["music"]))


def build(source: str, config_raw: bytes) -> str:
    cfg = json.loads(config_raw)
    cfg["_sha256"] = hashlib.sha256(config_raw).hexdigest()
    text = source
    for anchor in (ANCHOR_HELPERS, ANCHOR_LOAD, ANCHOR_INIT, ANCHOR_FILE, ANCHOR_PUBLISH, ANCHOR_SCORE):
        if text.count(anchor) != 1:
            raise SystemExit("anchor count mismatch: " + anchor.strip()[:80])
    text = text.replace(ANCHOR_HELPERS, helpers(cfg) + ANCHOR_HELPERS)
    text = text.replace(ANCHOR_LOAD, LOAD + ANCHOR_LOAD)
    text = text.replace(ANCHOR_INIT, ANCHOR_INIT + INIT)
    text = text.replace(ANCHOR_SCORE, ANCHOR_SCORE + SCORE)
    text = text.replace(ANCHOR_FILE, FILE + ANCHOR_FILE)
    text = text.replace(ANCHOR_PUBLISH, ANCHOR_PUBLISH + PUBLISH)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "submission/script_sub100.py")
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "submission/script_sub102.py")
    a = ap.parse_args()
    src = a.source.read_bytes()
    if hashlib.sha256(src).hexdigest() != a.source_sha256:
        raise SystemExit("source runner SHA mismatch")
    out = build(src.decode("utf-8"), a.config.read_bytes()).encode("utf-8")
    compile(out, "script_sub102.py", "exec")
    a.out.write_bytes(out)
    print(hashlib.sha256(out).hexdigest())


if __name__ == "__main__":
    main()
