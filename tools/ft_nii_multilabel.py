"""sub102: anchored multi-label fine-tune of the NII AntiDeepfake backbone on raw-mixture windows.

Heads (one linear layer over the mean-pooled final hidden state, like proj_fc):
  0 FILE-fake, 1 VOICE-fake, 2 MUSIC-fake, 3 VOICE-present, 4 MUSIC-present.
The three fake heads start at the pretrained <fake - real> direction, so step 0 equals
the pretrained NII fake logit. Window labels come from the exact render layout
(mixture_v2/sub102_items.jsonl); VOICE/MUSIC-fake losses are masked where that
component is absent or ambiguous. Anchor: decoupled L2-SP pull toward the pretrained
weights after each AdamW step (no decay toward zero), low LR, lower layers frozen.

Deploy grid: 4 s windows with a 2 s hop (script_sub100._sub80_window_starts), each window
normalized like predict_nii_fake, scored under bf16 autocast.

  .venv/Scripts/python.exe tools/ft_nii_multilabel.py --out $DATA_DIR/runs/sub102-nii-ml/r1
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ft_antideepfake as ft  # noqa: E402

SR = 16_000
WIN = 4 * SR
HOP = 2 * SR
MAX_WINDOWS = 150
BATCH = 16
HEADS = ("file", "voice", "music", "vp", "mp")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
ITEMS = DATA_DIR / "derived/mixture_v2/sub102_items.jsonl"
BASE = DATA_DIR / "models/nii_antideepfake/model.safetensors"
BASE_SHA = "b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5"
LEASE = ROOT / "scratchpad/gpu_leases/deepvoice-sub102-ft.json"
REQUEST = ROOT / "scratchpad/gpu.request"
PAUSE_EXIT = 3


def window_starts(n, width=WIN, hop=HOP):
    """Same as script_sub100._sub80_window_starts."""
    if n <= width:
        return [0]
    last = n - width
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def window_labels(item, start, width=WIN):
    """Targets and masks for window [start, start+width) from the component intervals.

    present: overlap >= 1 s -> 1, none -> 0, else masked.
    fake: fake overlap >= 0.5 s -> 1; present with no fake overlap -> 0; else masked.
    FILE: any fake -> 1; everything present known-real -> 0; else masked."""
    end = start + width
    y = np.zeros(5, np.float32)
    m = np.zeros(5, np.float32)
    fake = {}
    for h, kind in ((1, "speech"), (2, "music")):
        ov = fk = 0
        for a, b, k, f in item["comps"]:
            if k != kind:
                continue
            o = max(0, min(end, b) - max(start, a))
            ov += o
            if f:
                fk += o
        present = 1 if ov >= SR else (0 if ov == 0 else None)
        if fk >= SR // 2:
            fake[h] = 1
        elif fk == 0 and present == 1:
            fake[h] = 0
        else:
            fake[h] = None
        if present == 0:
            fake[h] = None
        ph = 3 if h == 1 else 4
        if present is not None:
            y[ph], m[ph] = present, 1
        if fake[h] is not None:
            y[h], m[h] = fake[h], 1
    # label-safety overrides from the item
    if item.get("source") == "sub99_view" and item.get("mask_music"):
        m[2] = m[4] = 0
    if item.get("kind") == "music_real":
        m[3] = 0  # FMA songs may contain undeclared vocals
    if fake[1] != 1 and fake[2] != 1 and item.get("source") == "sub99_view" and item.get("file_fake") is None:
        m[0] = 0
    elif fake[1] == 1 or fake[2] == 1:
        y[0], m[0] = 1, 1
    elif not any(f and max(0, min(end, b) - max(start, a)) > 0 for a, b, k, f in item["comps"]):
        y[0], m[0] = 0, 1  # no fake material in the window at all (a sub-threshold fake overlap stays masked)
    return y, m


def read_window(path, start, n_total, width=WIN):
    import soundfile as sf
    info = sf.info(str(path))
    if info.samplerate == SR and info.channels >= 1:
        data, _ = sf.read(str(path), start=start, frames=min(width, max(0, info.frames - start)),
                          dtype="float32", always_2d=True)
        x = data.mean(axis=1)
    else:
        x = ft.load_audio(path)[start:start + width]
    if x.size < width:
        x = np.tile(x, width // max(1, x.size) + 1)[:width] if x.size else np.zeros(width, np.float32)
    return x.astype(np.float32)


class WindowDataset:
    """Infinite random-window sampler (mixture renders vs views by share)."""

    def __init__(self, items, steps, batch, seed, view_share=0.25, aug=True):
        self.mix = [i for i in items if i["source"] == "mixture_v2"]
        self.views = [i for i in items if i["source"] == "sub99_view"]
        self.n = steps * batch
        self.seed = seed
        self.view_share = view_share if self.views else 0.0
        self.aug = ft.AugConfig(p_gain=0.3, p_noise=0.2, noise_snr_db=(15.0, 40.0), p_mp3=0.1) if aug else None

    def __len__(self):
        return self.n - getattr(self, "start", 0)

    def __getitem__(self, idx):
        idx += getattr(self, "start", 0)  # mid-epoch resume skips the already-seen draws
        rng = np.random.default_rng([self.seed, idx])
        pool = self.views if rng.random() < self.view_share else self.mix
        item = pool[int(rng.integers(len(pool)))]
        n = item["n"]
        start = int(rng.integers(0, max(1, n - WIN) + 1)) if n > WIN else 0
        x = read_window(item["path"], start, n)
        y, m = window_labels(item, start, min(WIN, n) if n < WIN else WIN)
        if self.aug is not None:
            x = ft.augment(x, self.aug, rng)
        x = ft.normalize(x)
        return x, y, m


class MultiHead:
    """Wrap the native detector: mean-pooled final hidden -> 5 logits."""

    @staticmethod
    def build(native, weight_path, torch):
        nn = torch.nn
        det = ft.build_model(native, weight_path)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.det = det
                self.heads = nn.Linear(det.proj_fc.in_features, len(HEADS))
                with torch.no_grad():
                    w = det.proj_fc.weight[0] - det.proj_fc.weight[1]
                    b = det.proj_fc.bias[0] - det.proj_fc.bias[1]
                    self.heads.weight.zero_()
                    self.heads.bias.zero_()
                    self.heads.weight[:3] = w
                    self.heads.bias[:3] = b

            def forward(self, wav):
                emb = self.det.m_ssl.extract_feat(wav)
                return self.heads(emb.mean(dim=1))

        return Net()


def masked_bce(logits, y, m, head_w):
    import torch
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    per_head = (loss * m).sum(0) / m.sum(0).clamp_min(1.0)
    return (per_head * head_w).sum(), per_head.detach()


def score_windows(model, audio, device, batch=BATCH, amp=True):
    """Runner-matched scoring: grid windows, per-window normalization, bf16 autocast -> (W,5) logits."""
    import torch
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    starts = window_starts(audio.size)
    if len(starts) > MAX_WINDOWS:  # runtime cap (> ~5 min audio): evenly spaced subset
        starts = [starts[i] for i in sorted(set(np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int).tolist()))]
    crops = [ft.normalize(audio[s:s + WIN]) for s in starts]
    out = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                enabled=amp and torch.device(device).type == "cuda"):
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            if len({c.size for c in chunk}) == 1:
                t = torch.from_numpy(np.stack(chunk)).to(device)
                out.append(model(t).float().cpu().numpy())
            else:
                for c in chunk:
                    out.append(model(torch.from_numpy(c)[None].to(device)).float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64), starts


def lse_pool(z, tau=1.0, w=None):
    z = np.asarray(z, np.float64)
    w = np.ones_like(z) if w is None else np.asarray(w, np.float64)
    w = w / max(w.sum(), 1e-12)
    zmax = z.max()
    return float(tau * np.log(np.sum(w * np.exp((z - zmax) / tau))) + zmax)


_AUDIO_CACHE = {}


def file_eval(model, items, device):
    """Per-file EER for FILE/VOICE/MUSIC with mean, max and LSE(tau=1) pooling."""
    from ft_antideepfake import binary_metrics
    per = []
    for it in items:
        audio = ft.load_audio(it["path"])  # on the fly (no cache: job RSS <= 5 GB)
        z, _ = score_windows(model, audio, device)
        per.append((it, z))
        del audio
    res = {}
    for h, name, mask_key in ((0, "file", None), (1, "voice", "voice_present"), (2, "music", "music_present")):
        sel = [(it, z) for it, z in per if mask_key is None or it.get(mask_key) == 1]
        labels = [it[f"{name}_fake"] for it, _ in sel]
        for pool in ("mean", "max", "lse1"):
            s = []
            for _, z in sel:
                v = z[:, h]
                p = v.mean() if pool == "mean" else v.max() if pool == "max" else lse_pool(v, 1.0)
                s.append(1 / (1 + math.exp(-p)))
            res[f"{name}_{pool}"] = binary_metrics(labels, s)["eer"]
    res["mean3_lse1"] = float(np.mean([res["file_lse1"], res["voice_lse1"], res["music_lse1"]]))
    return res, per


def rss_text():
    try:
        import psutil
        procs = [psutil.Process()] + psutil.Process().children(recursive=True)
        info = [pr.memory_info() for pr in procs]
        return (f"rss {sum(i.rss for i in info) / 2**30:.2f}GB private "
                f"{sum(getattr(i, 'private', 0) for i in info) / 2**30:.2f}GB (main {info[0].rss / 2**30:.2f}/"
                f"{getattr(info[0], 'private', 0) / 2**30:.2f})")
    except Exception as error:
        return f"rss ? ({error})"


def write_lease(step, cap_gb):
    LEASE.parent.mkdir(parents=True, exist_ok=True)
    LEASE.write_text(json.dumps(dict(project="deepvoice", job="sub102-nii-ft", cap_gb=cap_gb, ram_gb=5,
                                     pid=os.getpid(), window="2026-09-27..28 (pauses on gpu.request)",
                                     checkpoint=f"step {step}", updated=time.strftime("%Y-%m-%dT%H:%M:%S"))),
                     encoding="utf-8")


def release_lease():
    try:
        if LEASE.exists() and json.loads(LEASE.read_text(encoding="utf-8")).get("pid") == os.getpid():
            LEASE.unlink()
    except Exception:
        pass


def save_resume(torch, net, opt, state, resume):
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    torch.save(dict(model={k: v for k, v in net.state_dict().items() if k in names}, opt=opt.state_dict(), state=state),
               resume.with_suffix(".tmp"))
    os.replace(resume.with_suffix(".tmp"), resume)


def trainable_state(net):
    return {k: v.detach().cpu().float().contiguous() for k, v in net.named_parameters() if v.requires_grad}


def load_initial_overlay(net, path, device):
    """Load an epoch overlay; newly unfrozen parameters retain pretrained values."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        if meta.get("base_sha", meta.get("base_sha256")) != BASE_SHA:
            raise ValueError("init overlay base sha mismatch")
        keys = set(handle.keys())
    if not {"heads.weight", "heads.bias"} <= keys:
        raise ValueError("init overlay must include the multi-label heads")
    unknown = keys - set(net.state_dict())
    if unknown:
        raise ValueError(f"unexpected init keys: {sorted(unknown)}")
    net.load_state_dict(load_file(str(path), device=str(device)), strict=False)
    print("INITIALIZED", str(path), "tensors", len(keys), flush=True)


def train(args):
    import torch
    from safetensors.torch import save_file, load_file
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if REQUEST.exists():
        print("GPU_REQUEST present at start; exiting for the queue to retry", flush=True)
        return PAUSE_EXIT
    if ft.sha256_file(BASE) != BASE_SHA:
        raise SystemExit("base sha mismatch")
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    train_items = [i for i in items if i["split"] == "train"]
    val_items = [i for i in items if i["split"] == "val"]  # model selection always on the v2 val split
    for extra in args.extra_items:  # e.g. mixture_v3 (train split only; its val/test stay out)
        more = [json.loads(l) for l in open(extra, encoding="utf-8")]
        more = [i for i in more if i["split"] == "train"]
        held = {g for i in items if i["split"] in ("val", "test") for g in i["group"].split("|")}
        leak = [i["id"] for i in more if set(i["group"].split("|")) & held]
        if leak:
            raise SystemExit(f"{extra}: {len(leak)} train items share v2 holdout source groups, e.g. {leak[:3]}")
        print("EXTRA_ITEMS", extra, len(more), flush=True)
        train_items += [dict(i, source="mixture_v2" if i["source"] != "sub99_view" else i["source"]) for i in more]
    if args.v2_repeat > 1:
        v2 = [i for i in items if i["split"] == "train" and i["source"] == "mixture_v2"]
        train_items += v2 * (args.v2_repeat - 1)
    if args.val_limit:
        rng = np.random.default_rng(0)
        val_items = [val_items[i] for i in sorted(rng.choice(len(val_items), args.val_limit, replace=False))]
    write_lease(0, args.cap_gb)  # visible in the GPU budget while waiting for VRAM
    ft.wait_for_resources(min_ram_gb=12, min_vram_gb=args.min_free_vram, interval_s=60, label=" sub102")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    frac = ft.set_vram_cap(torch, device, args.cap_gb)
    native = ft.load_frontend()
    net = MultiHead.build(native, BASE, torch)
    info = ft.freeze(net.det, freeze_cnn=True, freeze_layers=args.freeze_layers)
    net.det.proj_fc.requires_grad_(False)
    ft.enable_grad_checkpointing(net.det)
    net.to(device)
    gc.collect()
    adt = torch.bfloat16 if args.anchor_bf16 else torch.float32
    anchors = {n: p.detach().clone().to(adt) for n, p in net.named_parameters() if p.requires_grad and not n.startswith("heads.")}
    head_params = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("heads.")]
    bb_params = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("heads.")]
    opt = torch.optim.AdamW([dict(params=bb_params, lr=args.lr, weight_decay=0.0),
                             dict(params=head_params, lr=args.lr_head, weight_decay=0.0)], betas=(0.9, 0.98),
                            foreach=False if args.anchor_bf16 else None)
    total = args.epochs * args.steps_per_epoch
    head_w = torch.tensor(args.head_weights, dtype=torch.float32, device=device)
    state = dict(epoch=0, step=0, history=[], best=None, bi=-1)
    resume = out / "resume.pt"
    # Anchors above remain the original pretrained weights, including on resume.
    if args.init and not resume.exists():
        load_initial_overlay(net, args.init, device)
        gc.collect()
        torch.cuda.empty_cache()
    if resume.exists():
        ck = torch.load(resume, map_location=device, weights_only=False)  # straight to GPU: no host RAM spike
        net.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        del ck
        gc.collect()
        torch.cuda.empty_cache()
        print("RESUMED", state["epoch"], state["step"], flush=True)
    write_lease(state["step"], args.cap_gb)
    try:
        if state["epoch"] == 0 and not state["history"]:
            t0 = time.time()
            net.eval()
            res, _ = file_eval(net, val_items, device)
            net.train()
            state["history"].append(dict(epoch=0, step=0, val=res, eval_s=time.time() - t0))
            print("EVAL epoch 0", json.dumps(res), flush=True)
        while state["epoch"] < args.epochs:
            ep = state["epoch"]
            ds = WindowDataset(train_items, args.steps_per_epoch, args.batch, seed=args.seed * 1000 + ep, view_share=args.view_share)
            first = state.get("bi", -1) + 1  # >0 only when resuming mid-epoch
            ds.start = first * args.batch
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                                 persistent_workers=False, drop_last=True, pin_memory=False)
            net.train()
            t0 = time.time()
            agg = np.zeros(5)
            nb = 0
            for bi, (x, y, m) in enumerate(loader, start=first):
                step = ep * args.steps_per_epoch + bi
                f = ft.lr_factor(step, args.warmup, total)
                opt.param_groups[0]["lr"] = args.lr * f
                opt.param_groups[1]["lr"] = args.lr_head * f
                x, y, m = x.to(device), y.to(device), m.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = net(x)
                loss, per_head = masked_bce(logits, y, m, head_w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bb_params + head_params, 1.0)
                opt.step()
                with torch.no_grad():  # decoupled L2-SP toward the pretrained weights
                    shrink = args.lr * f * args.sp
                    for n, p in net.named_parameters():
                        if n in anchors:
                            p.sub_(shrink * (p - anchors[n].to(p.dtype)))
                agg += per_head.cpu().numpy()
                nb += 1
                if bi % 100 == 0:
                    drift = math.sqrt(sum(float(((p - anchors[n].to(p.dtype)) ** 2).sum()) for n, p in net.named_parameters() if n in anchors))
                    print(f"ep {ep} step {bi}/{args.steps_per_epoch} loss {float(loss):.4f} heads "
                          f"{np.round(agg / nb, 4).tolist()} drift {drift:.4f} {(time.time() - t0) / max(nb, 1):.3f}s/step "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB {rss_text()}", flush=True)
                if args.ckpt_every and bi and bi % args.ckpt_every == 0 and bi < args.steps_per_epoch - 1:
                    state["bi"] = bi
                    save_resume(torch, net, opt, state, resume)
                    write_lease(ep * args.steps_per_epoch + bi, args.cap_gb)
                    if args.deadline and datetime.now() >= datetime.fromisoformat(args.deadline):
                        (out / "STOPPED").write_text("deadline; resume.pt preserved", encoding="utf-8")
                        print("DEADLINE_STOP", flush=True)
                        return 4
                    if REQUEST.exists():
                        print("GPU_REQUEST seen at mid-epoch checkpoint; pausing", flush=True)
                        return PAUSE_EXIT
            # end of epoch: eval + checkpoint
            net.eval()
            te = time.time()
            res, _ = file_eval(net, val_items, device)
            net.train()
            rec = dict(epoch=ep + 1, step=(ep + 1) * args.steps_per_epoch, train_heads=(agg / max(nb, 1)).tolist(),
                       val=res, train_s=te - t0, eval_s=time.time() - te,
                       s_per_step=(te - t0) / max(nb, 1), peak_gb=torch.cuda.max_memory_allocated() / 2**30)
            state["history"].append(rec)
            state["epoch"] = ep + 1
            state["step"] = rec["step"]
            state["bi"] = -1
            print("EVAL", json.dumps(rec), flush=True)
            st = trainable_state(net)
            save_file(st, str(out / f"epoch{ep + 1:02d}.safetensors"), metadata=dict(base_sha=BASE_SHA, epoch=str(ep + 1)))
            score = res["mean3_lse1"]
            if state["best"] is None or score < state["best"]["score"]:
                state["best"] = dict(epoch=ep + 1, score=score)
            save_resume(torch, net, opt, state, resume)
            (out / "history.json").write_text(json.dumps(dict(args=vars(args), freeze=info, cap_fraction=frac,
                                                              state=state), indent=1), encoding="utf-8")
            write_lease(state["step"], args.cap_gb)
            if args.regression_baseline is not None and state["epoch"] >= 3:
                recent = state["history"][-3:]
                if all(r["val"]["mean3_lse1"] > args.regression_baseline + args.regression_margin for r in recent):
                    (out / "STOPPED").write_text(json.dumps(dict(reason="three-epoch regression", best=state["best"])), encoding="utf-8")
                    print("REGRESSION_STOP", json.dumps(state["best"]), flush=True)
                    return 4
            if REQUEST.exists():
                print("GPU_REQUEST seen at checkpoint; pausing", flush=True)
                return PAUSE_EXIT
            if args.patience and state["epoch"] - state["best"]["epoch"] >= args.patience:
                print("EARLY_STOP", flush=True)
                break
        (out / "DONE").write_text(json.dumps(state["best"]), encoding="utf-8")
        return 0
    finally:
        release_lease()


def main(argv=None):
    global LEASE
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--init", help="warm-start epoch safetensors; existing resume.pt takes precedence")
    p.add_argument("--lease", default=str(LEASE), help="job-specific GPU lease path")
    p.add_argument("--deadline", help="local ISO time; stop at the next checkpoint")
    p.add_argument("--regression-baseline", type=float, help="stop after 3 consecutive epochs clearly worse than this EER")
    p.add_argument("--regression-margin", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=2500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--sp", type=float, default=5.0, help="decoupled L2-SP strength (per-step shrink lr*f*sp)")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--freeze-layers", type=int, default=8)
    p.add_argument("--head-weights", type=float, nargs=5, default=[1.0, 1.0, 1.0, 0.2, 0.2])
    p.add_argument("--view-share", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cap-gb", type=float, default=6.0)
    p.add_argument("--min-free-vram", type=float, default=6.5)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--extra-items", nargs="*", default=[])
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--v2-repeat", type=int, default=1)
    p.add_argument("--anchor-bf16", action="store_true", help="bf16 anchors + non-foreach AdamW (lower VRAM)")
    args = p.parse_args(argv)
    if args.deadline:
        datetime.fromisoformat(args.deadline)
    LEASE = Path(args.lease)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
