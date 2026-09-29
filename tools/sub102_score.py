"""sub102: score runner-matched raw-mixture windows with a fine-tuned NII multi-label checkpoint.

Sets: holdout (mixture_v2 val+test renders), panels (korean_speech, korean_mix, external, song via
direct_head_eval.panel_rows). Output jsonl per set: key, split/panel, labels, starts, logits (W x 5).

  .venv/Scripts/python.exe tools/sub102_score.py --ckpt $DATA_DIR/r1/epoch03.safetensors --sets holdout panels --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ft_antideepfake as ft  # noqa: E402
import ft_nii_multilabel as ml  # noqa: E402


V3_ITEMS = str(Path(os.environ.get("DATA_DIR", "data")) / "derived/mixture_v2/sub102_items_v3.jsonl")


def build(ckpt, device):
    import torch
    from safetensors.torch import load_file
    native = ft.load_frontend()
    net = ml.MultiHead.build(native, ml.BASE, torch)
    if ckpt:
        state = load_file(str(ckpt))
        missing = [k for k in state if k not in dict(net.named_parameters())]
        if missing:
            raise KeyError(missing[:5])
        net.load_state_dict(state, strict=False)
    return net.to(device).eval()


def holdout_rows(path=None):
    items = [json.loads(l) for l in open(path or ml.ITEMS, encoding="utf-8")]
    return [dict(key=i["id"], path=i["path"], split=i["split"], kind=i["kind"], channel=i["channel"],
                 file_label=i["file_fake"], voice_label=i["voice_fake"], music_label=i["music_fake"],
                 voice_present=i["voice_present"], music_present=i["music_present"])
            for i in items if i["source"] in ("mixture_v2", "mixture_v3") and i["split"] in ("val", "test")]


def panel_rows():
    import direct_head_eval as ev
    out = []
    for panel in ("korean_speech", "korean_mix", "external", "song"):
        for r in ev.panel_rows(panel):
            def lab(v):
                return int(float(v)) if str(v).strip() not in ("", "nan", "None") else None
            out.append(dict(key=f"{panel}|{r['sample_id']}", panel=panel, path=r["audio_path"],
                            file_label=lab(r["file_label"]), voice_label=lab(r["voice_label"]),
                            music_label=lab(r["music_label"]), voice_present=lab(r["voice_present"]),
                            music_present=lab(r["music_present"])))
    return out


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--sets", nargs="+", default=["holdout", "panels"])
    p.add_argument("--out", required=True)
    p.add_argument("--cap-gb", type=float, default=6.0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    ft.set_vram_cap(torch, device, args.cap_gb)
    net = build(args.ckpt, device)
    for name in args.sets:
        rows = (holdout_rows() if name == "holdout" else panel_rows() if name == "panels"
                else holdout_rows(V3_ITEMS))
        path = out / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")}
        t0 = time.time()
        nwin = 0
        with open(path, "a", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r["key"] in done:
                    continue
                try:
                    audio = ft.load_audio(r["path"])
                    z, starts = ml.score_windows(net, audio, device)
                    rec = dict(r, starts=[int(s) for s in starts], n=int(audio.size), logits=np.round(z, 5).tolist())
                except Exception as error:  # record and continue; eval treats it as a runner fallback
                    rec = dict(r, error=str(error))
                    z = np.zeros((0, 5))
                nwin += len(z)
                f.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    f.flush()
                    print(name, i, len(rows), f"{time.time() - t0:.1f}s windows {nwin}", flush=True)
        print("DONE", name, len(rows), f"{time.time() - t0:.1f}s windows {nwin}", flush=True)


if __name__ == "__main__":
    main()
