# Curated source report

Historical research report retained for technical details. Local paths are portable placeholders; internal operations lines were omitted. Later official results and final selection are documented in `REPORT.md` and `docs/experiments.md`.

# sub102: anchored multi-label NII fine-tune on raw-mixture windows (swing #2)

Status: COMPLETE — sub102 and sub102x built, validated, smoked and benchmarked locally.

## Setup
- Backbone: the deployed NII AntiDeepfake wav2vec-large (`nii_model.safetensors`, sha b27943fe…), native frontend (sha 80bc3a23…).
- Heads: one linear layer over the mean-pooled final hidden state, 5 outputs: FILE-fake, VOICE-fake, MUSIC-fake, VOICE-present, MUSIC-present. The three fake heads start at the pretrained `<fake - real>` direction, so step 0 is the pretrained NII fake logit.
- Anchor: CNN + layer_norm/post_extract_proj/pos_conv + layers 0-7 frozen (201.5M of 315.4M trainable); decoupled L2-SP pull toward the pretrained weights after each AdamW step (`p -= lr*f*5*(p-p0)`, no decay toward 0); LR 1e-5 backbone / 5e-4 head, 300-step warmup, cosine to 0.1x; bf16 autocast; per-layer gradient checkpointing; grad clip 1.
- Data (`sub102_items.jsonl`, `sub102_prep.py`): 4,800 mixture_v2 train renders + 2,016 sub99 label-safe speech views (25% of draws). Holdout: v2 val (600) and test (600) renders; source groups are disjoint across splits (2,328/287/303, 0 shared).
- Window labels come from the exact render layout. seq cut and order, and partial splice spans, are recorded. Overlap order was not recorded, so it was recovered by replaying the render RNG (verified on 40 clean renders, max |diff| 1.5e-5 = PCM16 quantization).
  - present: overlap >= 1 s -> 1, none -> 0, else masked.
  - fake: fake overlap >= 0.5 s -> 1; present with no fake -> 0; else masked. VOICE/MUSIC-fake losses are masked where that component is absent.
  - FILE: any fake -> 1; no fake material in the window -> 0; a sub-threshold fake overlap is masked. Real-only windows of fake files are labelled 0.
  - Views: VOICE = speech label; FILE only where the view's manifest makes it safe; MUSIC is masked where unknown.
- Training windows: random 4 s crops (soundfile partial reads), light augmentation (gain, noise, MP3), per-window normalization exactly as predict_nii_fake.
- Deploy grid: 4 s windows with a 2 s hop (`_sub80_window_starts`), at most 150 windows per file (evenly spaced), batch 16, bf16 autocast.

## r1 (8 x 2,500 steps, batch 8)
- 0.21-0.29 s/step; peak 4.96 GB allocated under a 6 GB cap; job RSS 4.4 GB after the 13:05 low-RAM restart (resumed from the epoch-2 checkpoint).
- Val (v2 val renders, standalone FT, per-file pooling) EERs:

| epoch | FILE mean/max/lse1 | VOICE mean/max/lse1 | MUSIC mean/max/lse1 |
|---|---|---|---|
| 0 (pretrained) | .355/.343/.337 | .303/.303/.303 | .454/.452/.450 |
| 1 | .190/.183/.172 | .155/.178/.169 | .228/.247/.251 |
| 2 | .162/.172/.168 | .155/.160/.150 | .199/.245/.237 |
| 3 | .143/.158/.147 | .139/.146/.141 | .170/.216/.205 |
| 4 | .128/.137/.135 | .120/.141/.139 | .176/.212/.203 |
| 5 | .153/.143/.138 | .120/.137/.130 | .156/.193/.189 |
| 6 | .138/.158/.150 | .139/.120/.116 | .160/.193/.185 |
| **7 (best)** | .143/.145/.143 | .120/.116/.116 | .152/.185/.180 |
| 8 | .138/.150/.143 | .116/.125/.116 | .160/.189/.185 |

- Two transient CUDA faults (illegal memory access, cuBLAS execution failure) on the crowded GPU; resumed from checkpoints both times. Since then: GPU-side resume, checkpoints every 500 steps, auto-retry in the queue.
- Exported deploy overlay (fp16, 260 tensors, 403 MB): `runs/sub102-nii-ml/r1/nii_ml_sub102_r1e07.safetensors`, sha256 d9f5e4ca705589e53273480ff3c730bf2adeee6eb84467bb83f3765972587b15.

## Holdout vs incumbent (r1 e07)
The incumbent is the exact sub100 output rebuilt from the sub101 replay records. The reconstruction was checked on 2,756 panel rows: VOICE equals sub99 A_free exactly and FILE equals the replayed floor exactly. The FT FILE probability is blended into the pre-floor FILE, and the floor is re-applied.
Pooling per head is a smooth max (tau), optionally presence-weighted (gamma), with an affine map. It is fit on one split and the blend weight is read on the other.

| head | test base | test best (w) | val base | val best (w) |
|---|---|---|---|---|
| FILE | .205 | .118 (.35-.4) | .198 | .095 (.75-1.0) |
| VOICE | .246 | .104 (.55-.65) | .188 | .088 (.55-.85) |
| MUSIC | .101 | .056 (.35-.4) | .108 | .062 (.45-.5) |

Standalone FT on test: FILE .152, VOICE .138, MUSIC .161.

## Panels (information only; r1 e07, pooling fit on v2 val)

| panel/head | sub100 | w .1 | w .2 | w .3 | w .5 | w 1.0 (FT alone, calibrated) |
|---|---|---|---|---|---|---|
| korean_speech FILE | .0547 | .0512 | .0512 | .0582 | .0746 | .1016 |
| korean_speech VOICE | .0486 | .0486 | .0469 | .0486 | .0660 | .1137 |
| korean_mix FILE | .1757 | .1642 | .1580 | .1580 | .1653 | .1966 |
| korean_mix VOICE | .1578 | .1547 | .1703 | .1750 | .1906 | .2672 |
| korean_mix MUSIC | .0938 | .0969 | .1000 | .1125 | .1406 | .3063 |
| external FILE | .1958 | .1786 | .1667 | .1732 | .1851 | .2089 |
| external VOICE | .1270 | .1190 | .1032 | .0992 | .1190 | .2024 |
| external MUSIC | .1429 | .1395 | .1429 | .1361 | .1769 | .2993 |
| song FILE | .1566 | .1566 | .1566 | .1566 | .1639 | .1882 |
| song VOICE | .2100 | .1940 | .1860 | .1860 | .1860 | .2180 |
| song MUSIC | .1380 | .1620 | .1540 | .1540 | .1700 | .2480 |

Reading: the holdout is in-distribution for the render generator and overstates transfer. On panels, FILE gains at w .2-.3 on every panel (song flat). VOICE helps only at w ~.1 (at .2 korean_mix worsens). MUSIC does not transfer (korean_mix and song worse; external flat).

## Deploy / build

Completed 2026-09-27T20:18:12 KST. Both packages are prepared locally; no DACON upload was performed.

### Evaluation and checkpoint choice

Scored the exported fp16 overlays of r2 epochs 10, 9 and 8, using `tools/sub102_score.py` and the unchanged `sub102_eval.py`, on 600 v2 val + 600 v2 test files and all 2,756 panel files. All four checkpoints (including retained r1 e07) have 3,956/3,956 incumbent coverage and zero scoring errors. Retained `sub101_features_*.jsonl` supply exact sub100 replay outputs.

r2 uses mixture_v2 plus mixture_v3 (including Korean), 10 epochs, CNN/frontend and transformer layers 0-3 frozen. Export: 324 tensors, 503,904,202 bytes per overlay. All pooling/affine parameters used for deployment are fit on v2 val; the test split is a local development holdout, not competition test data.

EER below is a fraction (lower is better). Panel pooling concatenates labeled records and evaluates one EER, rather than averaging panel EERs. FILE/VOICE weight choice minimizes pooled EER; MUSIC minimizes 0.6 × external EER + 0.4 × pooled EER. Holdout test breaks ties. Checkpoint choice minimizes the equal-head mean of those objectives. Korean-mix gains under 1 pp are not independent positive evidence.

| checkpoint | standalone test FILE | VOICE | MUSIC | panel weights F/V/M | pooled FILE | VOICE | MUSIC | external MUSIC | selection objective |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| r1e07 | 0.151664 | 0.137724 | 0.160834 | 0.25/0.10/0.15 | 0.115000 | 0.140740 | 0.114070 | 0.139456 | 0.128347 |
| r2e10 | 0.118390 | 0.096663 | 0.125105 | 0.20/0.15/0.20 | 0.112525 | 0.137711 | 0.112238 | 0.136054 | 0.125588 |
| r2e09 | 0.129997 | 0.101510 | 0.106455 | 0.20/0.25/0.25 | 0.116105 | 0.137711 | 0.112238 | 0.129252 | 0.125421 |
| r2e08 | 0.133350 | 0.108641 | 0.125105 | 0.15/0.20/0.20 | 0.114735 | 0.138468 | 0.110983 | 0.132653 | 0.125729 |

**Selected checkpoint: r2e09 for both packages.** r2 transfers better under the declared panel objective; r1 e07 was not used.

**sub102 (panel-best):** {"file": 0.2, "music": 0.25, "voice": 0.25}.

**sub102x (aggressive):** {"file": 0.3, "music": 0.25, "voice": 0.35}. Larger weights minimize v2 test EER subject to every labeled individual panel staying within +1.5 pp of sub100. Exact holdout ties favor the larger weight. The complete admissible grid is in `sub102_selection.json`.

### Exact chosen outputs versus sub100

| split/panel | head | n | sub100 EER | sub102 EER | delta pp | sub102x EER | delta pp |
|---|---|---:|---:|---:|---:|---:|---:|
| pooled | file | 2756 | 0.125475 | 0.116105 | -0.937 | 0.119420 | -0.605 |
| pooled | voice | 2672 | 0.144445 | 0.137711 | -0.673 | 0.141078 | -0.337 |
| pooled | music | 1604 | 0.124686 | 0.112238 | -1.245 | 0.112238 | -1.245 |
| korean_speech | file | 1152 | 0.054688 | 0.056445 | +0.176 | 0.062500 | +0.781 |
| korean_speech | voice | 1152 | 0.048633 | 0.045117 | -0.352 | 0.060742 | +1.211 |
| korean_mix | file | 640 | 0.175745 | 0.157952 | -1.779 | 0.169468 | -0.628 |
| korean_mix | voice | 640 | 0.157816 | 0.157816 | +0.000 | 0.167215 | +0.940 |
| korean_mix | music | 640 | 0.093750 | 0.096875 | +0.313 | 0.096875 | +0.313 |
| external | file | 588 | 0.195833 | 0.173214 | -2.262 | 0.185119 | -1.071 |
| external | voice | 504 | 0.126984 | 0.107143 | -1.984 | 0.107143 | -1.984 |
| external | music | 588 | 0.142857 | 0.129252 | -1.361 | 0.129252 | -1.361 |
| song | file | 376 | 0.156613 | 0.153706 | -0.291 | 0.129360 | -2.725 |
| song | voice | 376 | 0.209997 | 0.185996 | -2.400 | 0.184012 | -2.599 |
| song | music | 376 | 0.137993 | 0.145993 | +0.800 | 0.145993 | +0.800 |
| val | file | 600 | 0.198289 | 0.109991 | -8.830 | 0.098380 | -9.991 |
| val | voice | 432 | 0.187536 | 0.101836 | -8.570 | 0.076425 | -11.111 |
| val | music | 518 | 0.108248 | 0.061750 | -4.650 | 0.061750 | -4.650 |
| test | file | 600 | 0.205055 | 0.126644 | -7.841 | 0.108331 | -9.672 |
| test | voice | 414 | 0.246365 | 0.142571 | -10.379 | 0.101510 | -14.485 |
| test | music | 535 | 0.101006 | 0.046626 | -5.438 | 0.046626 | -5.438 |

FILE is logit-blended into the sub100 pre-floor FILE; its original floor is reapplied using the original VOICE38 and MUSIC68 inputs. VOICE and MUSIC blends are published afterward. Both presence outputs are unchanged. There is no cross-file normalization, adaptation, or competition-test tuning.

### Artifacts, smoke and runtime

**sub102**

- ZIP: `submissions/sub102.zip`; 8,587,151,130 bytes (<10 GB).
- ZIP SHA256: `f3644b71b76d787a2e9b9746d3e12d0ee5825d5adb91937cbc140fac0eb5a57b`.
- Runner SHA256: `1df11f51e7d370ad7e553d2781233a4d38c3c78b54978ab2e04fd1547db4639f`.
- Overlay SHA256: `afd2679ac81c12c91360271e35b057222dc533bfe5cca72dda669d8569ba6129`.
- Source revision: `3310b60892e83d210b308b83b1e3d489e14b0145`; uncommitted candidate source is bound by exact runner/config/generator/build-tool digests in the build receipt.
- CPU tests: 5/5 passed; exact generated-runner identity passed. `tools/validate_submission.py` passed (58 ZIP members).
- Runner versus evaluator GPU parity: 24 files; max absolute window-logit difference 0.0.
- Offline smoke: normal and reversed each passed, zero network attempts, no healthy-input warnings, exact order independence, neutral corrupt-file fallback, unchanged presence columns. Post-smoke ZIP and projected-member hashes verified.

| order | files | seconds | FILE changed | VOICE changed | MUSIC changed | peak total GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| normal | 22 | 281.62 | 20 | 21 | 21 | 16025 |
| reversed | 22 | 56.87 | 20 | 21 | 21 | 14918 |

Added-pass runtime bench on the actual runner (150 stratified files; CUDA-synchronized, 5 GiB cap): mean excluding first 0.0315s/file, p95 0.0459s, total 4.96s, overlay load 0.85s, peak Torch allocation 3.22 GiB. Anchored to the known ~40-minute sub100 official runtime, adding 1,200 such forwards projects **40.64 min**. The conservative bound is **48.77 min** (maximum of +20% on the entire projection, or doubled added-pass mean/p95), below 60 min. This is a projection, not a fresh official L4 measurement.

**sub102x**

- ZIP: `submissions/sub102x.zip`; 8,587,151,141 bytes (<10 GB).
- ZIP SHA256: `594feee876cfef60e1f7993e22f6a5ad616928c9906f51e740b037f676d2d3ea`.
- Runner SHA256: `f43d242c48796c88ad39476e20ccaadef9e9fb48fa948ccb34d4d3e868f1f566`.
- Overlay SHA256: `afd2679ac81c12c91360271e35b057222dc533bfe5cca72dda669d8569ba6129`.
- Source revision: `3310b60892e83d210b308b83b1e3d489e14b0145`; uncommitted candidate source is bound by exact runner/config/generator/build-tool digests in the build receipt.
- CPU tests: 5/5 passed; exact generated-runner identity passed. `tools/validate_submission.py` passed (58 ZIP members).
- Runner versus evaluator GPU parity: 24 files; max absolute window-logit difference 0.0.
- Offline smoke: normal and reversed each passed, zero network attempts, no healthy-input warnings, exact order independence, neutral corrupt-file fallback, unchanged presence columns. Post-smoke ZIP and projected-member hashes verified.

| order | files | seconds | FILE changed | VOICE changed | MUSIC changed | peak total GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| normal | 22 | 49.76 | 20 | 21 | 21 | 16002 |
| reversed | 22 | 41.76 | 20 | 21 | 21 | 14954 |

Added-pass runtime bench on the actual runner (150 stratified files; CUDA-synchronized, 5 GiB cap): mean excluding first 0.0314s/file, p95 0.0458s, total 4.91s, overlay load 0.73s, peak Torch allocation 3.22 GiB. Anchored to the known ~40-minute sub100 official runtime, adding 1,200 such forwards projects **40.64 min**. The conservative bound is **48.77 min** (maximum of +20% on the entire projection, or doubled added-pass mean/p95), below 60 min. This is a projection, not a fresh official L4 measurement.

### Risks and retained evidence

- Holdout blends substantially overstate observed panel transfer. The aggressive package intentionally trades some panel EER for stronger local mixture holdout performance; neither candidate establishes a .89 leaderboard score.
- Panels were reused for checkpoint/weight selection, so their improvements are selection-biased. MUSIC external weighting reflects prior transfer evidence but does not guarantee a leaderboard gain. The r2 e09 versus e10 selection-objective gap is only about 0.017 pp; this is a narrow development-set choice, not evidence of a robust epoch advantage.
- Shared-GPU timing and the 150-file duration mix may differ from official evaluation. Runtime acceptance requires the conservative added-pass projection to remain under 60 minutes. A full-stack source bench completed in 379.49 s, but consecutive 20-second files changed from ~1.7 s to 59-68 s under changing shared-GPU pressure. Its scaling ratio is invalid; the candidate full-stack attempt was stopped without changing either ZIP. Logs, GPU samples and stop evidence are retained in `sub102_source.runtime.json` and `sub102.contended_bench.json`. The added-pass benchmark repeats the established r1 measurement method with a 5 GiB cap; official L4 22.4 GiB hardware is documented in `docs/competition_ground_truth.md`.
- The sub102 normal smoke encountered a video job arriving after preflight and ran slowly under memory pressure; its reversed smoke ran after contention cleared. Both correctness checks passed. These smoke timings are not used for the official-runtime projection.
- Immutable `submissions/sub100.zip` was hash-checked before build and after smoke; retained original SHA256 `fcb20c2d2914903212f4094e48ed258f2577a2f5b76252151902d35a55581934`. Existing ZIPs, checkpoints, replay features and unrelated files were preserved.
- Complete 0.05-step curves, coverage and pooling parameters: `sub102_epoch_comparison.json`. Exact selection rationale and admissible aggressive weights: `sub102_selection.json`. Per-epoch canonical reports: `runs/sub102-nii-ml/r2/eval_e10.json`, `eval_e09.json`, `eval_e08.json`.
- Package bindings: `submissions/sub102.build.json`, `submissions/sub102x.build.json`; final acceptance: corresponding `.validation.json`; CPU, parity, archive validation, resources and runtime receipts alongside. Detached job logs are in `runs/sub102-nii-ml/r2/`.


## r3

Completed 2026-09-28T04:07:52 KST. **GO**. No upload performed.

Exported and scored r3 e12/e11/e10 with the unchanged sub102 exporter, scorer and canonical evaluator, using fp16 deploy overlays, 600 v2 val + 600 v2 test files and all 2,756 panel files. All pooling/affine fits use v2 val. Every epoch has 3,956/3,956 incumbent coverage and zero scoring errors. r2 e09 was recomputed from its retained scores; its selection record matches exactly.

FILE/VOICE: pooled-panel EER. MUSIC: 60% external EER + 40% pooled-panel EER. Holdout test tie-break. Epoch: equal-head mean objective. Korean-mix sub-1pp gains are not used as independent positive evidence.

| checkpoint | standalone test FILE | VOICE | MUSIC | weights F/V/M | pooled FILE | VOICE | MUSIC | external MUSIC | selection objective |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| r2e09 | 0.129997 | 0.101510 | 0.106455 | 0.20/0.25/0.25 | 0.116105 | 0.137711 | 0.112238 | 0.129252 | 0.125421 |
| r3e12 | 0.104978 | 0.096663 | 0.078479 | 0.15/0.10/0.35 | 0.114735 | 0.142593 | 0.114070 | 0.119048 | 0.124795 |
| r3e11 | 0.104978 | 0.101510 | 0.101006 | 0.15/0.10/0.30 | 0.115840 | 0.143350 | 0.108473 | 0.122449 | 0.125350 |
| r3e10 | 0.111684 | 0.101510 | 0.093253 | 0.15/0.10/0.35 | 0.120525 | 0.143350 | 0.114070 | 0.115646 | 0.126297 |

Selected: **r3e12**. Panel weights: `{"file": 0.15, "voice": 0.1, "music": 0.35}`.
Aggressive rule: team r3 task: largest grid weight at least panel-best, keeping every labeled panel within +1.5pp of sub100. This differs from the prior r2 holdout-minimizing aggressive rule.
Aggressive weights: `{"file": 0.4, "voice": 0.3, "music": 0.35}`.

| split/panel | head | sub100 EER | panel-best EER | aggressive EER |
|---|---|---:|---:|---:|
| pooled | file | 0.125475 | 0.114735 | 0.124105 |
| pooled | voice | 0.144445 | 0.142593 | 0.153451 |
| pooled | music | 0.124686 | 0.114070 | 0.114070 |
| korean_speech | file | 0.054688 | 0.050391 | 0.068555 |
| korean_speech | voice | 0.048633 | 0.048633 | 0.051172 |
| korean_mix | file | 0.175745 | 0.157952 | 0.157952 |
| korean_mix | voice | 0.157816 | 0.154684 | 0.167215 |
| korean_mix | music | 0.093750 | 0.093750 | 0.093750 |
| external | file | 0.195833 | 0.166667 | 0.173214 |
| external | voice | 0.126984 | 0.107143 | 0.103175 |
| external | music | 0.142857 | 0.119048 | 0.119048 |
| song | file | 0.156613 | 0.127907 | 0.130814 |
| song | voice | 0.209997 | 0.185996 | 0.184012 |
| song | music | 0.137993 | 0.152010 | 0.152010 |
| val | file | 0.198289 | 0.120073 | 0.073327 |
| val | voice | 0.187536 | 0.150435 | 0.062512 |
| val | music | 0.108248 | 0.042500 | 0.042500 |
| test | file | 0.205055 | 0.136704 | 0.086665 |
| test | voice | 0.246365 | 0.195609 | 0.091816 |
| test | music | 0.101006 | 0.037301 | 0.037301 |

### sub102r3

- ZIP: `submissions/sub102r3.zip`; 8,697,568,372 bytes; SHA256 `8accfe063952f34a0f0aaa782693c7931cf42f3a86a410a2eb4f0bd9fdd32250`.
- Runner SHA256 `c34db2573ba3fa7c4c4cad18fd4bbcd862cd9b7fe38f923255c4db2a9e15faff`; overlay SHA256 `6818a0ffbadd1fa66afa1ee882483b247b0963dda208f05ba3117642c9608439`.
- Source revision `3310b60892e83d210b308b83b1e3d489e14b0145`; exact runner/config/generator/build-tool digests bind the uncommitted candidate.
- CPU tests 5/5 passed; unchanged generator identity and archive validation passed. Runner/scorer parity: 24 files, max absolute logit difference 0.0.
- Offline normal and reversed smoke passed, with zero network attempts, unchanged presence outputs, neutral corrupt-file fallback, no healthy warnings, exact order independence, and verified post-smoke ZIP/member hashes.
- Smoke seconds normal/reversed: 179.20/71.51.
- Added pass on 150 files: mean excluding first 0.04469s, p95 0.05690s; overlay load 1.50s, peak Torch allocation 3.22 GiB. Projected official runtime 40.92 min; conservative 49.10 min (<60).

### sub102r3x

- ZIP: `submissions/sub102r3x.zip`; 8,697,568,367 bytes; SHA256 `0ff459bda5af4667d407ec01de9dac2421ff86fc5fd5903fb41ee06a24ec8383`.
- Runner SHA256 `eda86269ff732fc1f2028beddfec082f6954344960b792daedc6c5b7d12e1d71`; overlay SHA256 `6818a0ffbadd1fa66afa1ee882483b247b0963dda208f05ba3117642c9608439`.
- Source revision `3310b60892e83d210b308b83b1e3d489e14b0145`; exact runner/config/generator/build-tool digests bind the uncommitted candidate.
- CPU tests 5/5 passed; unchanged generator identity and archive validation passed. Runner/scorer parity: 24 files, max absolute logit difference 0.0.
- Offline normal and reversed smoke passed, with zero network attempts, unchanged presence outputs, neutral corrupt-file fallback, no healthy warnings, exact order independence, and verified post-smoke ZIP/member hashes.
- Smoke seconds normal/reversed: 71.55/70.16.
- Added pass on 150 files: mean excluding first 0.03921s, p95 0.05009s; overlay load 1.44s, peak Torch allocation 3.22 GiB. Projected official runtime 40.81 min; conservative 48.97 min (<60).

### r3 risks and provenance

- Reused panels and local mixture holdouts are selection-biased; lower EER here does not establish a leaderboard improvement. No competition-test tuning or cross-file inference was used.
- The r3 training note clarifies that freeze-layers=0 unfreezes all transformer layers and frontend normalization/projection/positional convolution; the convolutional feature extractor remains frozen.
- Runtime is a shared-GPU added-pass projection anchored to the prior ~40-minute sub100 official run, not a new official L4 measurement.
- Immutable sub102/r2 and sub100 artifacts were preserved. Complete curves, selections, scores, canonical reports and logs: `sub102r3_epoch_comparison.json`, `sub102r3_selection.json`, and `runs/sub102-nii-ml/r3/`. Candidate build/validation receipts are under `submissions/`.


### r3 interpretation and final audit

The e12 objective improvement over sub102/r2 e09 is 0.062599 percentage points. It comes from FILE and the external-weighted MUSIC objective. Pooled VOICE EER is worse than r2 e09 (0.142593 versus 0.137711). Pooled MUSIC also worsens (0.114070 versus 0.112238), while external MUSIC improves (0.119048 versus 0.129252). This is a narrow reused-panel selection result, not robust evidence of an official gain. The largest-weight alternative trades pooled performance for stronger mixture-holdout results. Its pooled VOICE EER is 0.153451 versus sub100 0.144445 even though each individual panel meets the constraint. Both candidates increase song MUSIC EER by 1.401690 pp versus sub100, near the 1.5 pp bound.

The initial detached launcher had a Windows standard-handle error before export; redirecting Python and subprocess output to explicit local log handles resolved it. All three exports, scoring runs and release checks subsequently completed. Final r3 audit and scoring/evaluator source digests are retained in `runs/sub102-nii-ml/r3/r3_final_audit.json`.


## sub105 ensemble

Completed 2026-09-28T04:16:34 KST. **NO-GO under the declared complexity guard; no package built and no GPU work or upload performed.**

team reports sub102 official score 0.8418682116 (ADS 0.825484127), versus sub100 0.8345967831. This is user-supplied official context, not a fresh website verification. The team reports that panel gains transferred much better than mixture-holdout gains. The declared selection objective is unchanged; no inferred ADS weighting was substituted.

The inputs are the retained fp16 r2 e09 and r3 e12 window scores for 600 mixture_v2 val + 600 test files and all 2,756 panel files. Keys, labels, presence masks, window starts, lengths and paths match exactly. Each model keeps its existing val-fitted pooling and affine calibration; no refit and no competition-test data were used.

### Frozen search and selection

- (a) Equal mean of calibrated per-head probabilities, or equal mean of their clipped logits; one blend weight per head into sub100, from the existing 0.00..1.00 grid in 0.05 steps. Choose the better complete mean family, rather than mixing families by head.
- (b) Direct convex logit combination: `(1-w2-w3)*logit(sub100) + w2*logit(r2) + w3*logit(r3)`, with each coefficient in `{0,.05,.10,.15,.20,.25,.30,.40,.50}` (81 pairs/head). No refinement or follow-up expansion.
- FILE is combined before the unchanged sub100 floor, which is reapplied using original presence/VOICE38/MUSIC68 inputs. Published VOICE and MUSIC are combined after that floor; presence columns do not change.
- FILE/VOICE minimize pooled-panel EER; MUSIC minimizes `.6*external EER + .4*pooled EER`. Holdout test is used only to break exact ties. Family objective is the equal-head mean. Separate model weights are admissible only when they beat the best mean family by **more than 0.3 pp**.

| candidate/family | FILE objective | VOICE objective | MUSIC objective | mean objective | head weights F/V/M |
|---|---:|---:|---:|---:|---|
| sub102 | 0.116105 | 0.137711 | 0.122446 | 0.125420773 | 0.2/0.25/0.25 |
| sub102r3 | 0.114735 | 0.142593 | 0.117057 | 0.124794788 | 0.15/0.1/0.35 |
| sub102r3x | 0.124105 | 0.153451 | 0.117057 | 0.131537498 | 0.4/0.3/0.35 |
| mean_probability | 0.115840 | 0.138468 | 0.120405 | 0.124904564 | 0.2/0.15/0.3 |
| mean_logit | 0.113895 | 0.139983 | 0.121138 | 0.125005477 | 0.2/0.1/0.3 |
| separate | 0.112791 | 0.137711 | 0.117863 | 0.122787977 | [0.15, 0.05]/[0.25, 0]/[0.05, 0.2] |

The best simple ensemble is **mean_probability**, objective **0.124904564**, which is **0.010978 pp worse** than sub102r3. Separate weights reach **0.122787977**, improving the simple ensemble by **0.211659 pp**, below the required >0.3 pp hurdle. Therefore the separate-weight result is retained as exploratory evidence and is not selected. No sub105 package, runner, build receipt or package-validation receipt was created.

### Per-panel and local holdout EERs

| group | head | n | sub102 | sub102r3 | sub102r3x | equal probabilities | equal logits | separate weights (ineligible) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | file | 2756 | 0.116105 | 0.114735 | 0.124105 | 0.115840 | 0.113895 | 0.112791 |
| pooled | voice | 2672 | 0.137711 | 0.142593 | 0.153451 | 0.138468 | 0.139983 | 0.137711 |
| pooled | music | 1604 | 0.112238 | 0.114070 | 0.114070 | 0.112238 | 0.114070 | 0.110983 |
| korean_speech | file | 1152 | 0.056445 | 0.050391 | 0.068555 | 0.056445 | 0.056445 | 0.058203 |
| korean_speech | voice | 1152 | 0.045117 | 0.048633 | 0.051172 | 0.045117 | 0.045117 | 0.045117 |
| korean_mix | file | 640 | 0.157952 | 0.157952 | 0.157952 | 0.157952 | 0.156915 | 0.156915 |
| korean_mix | voice | 640 | 0.157816 | 0.154684 | 0.167215 | 0.154684 | 0.151551 | 0.157816 |
| korean_mix | music | 640 | 0.096875 | 0.093750 | 0.093750 | 0.087500 | 0.084375 | 0.090625 |
| external | file | 588 | 0.173214 | 0.166667 | 0.173214 | 0.161310 | 0.165476 | 0.167857 |
| external | voice | 504 | 0.107143 | 0.107143 | 0.103175 | 0.103175 | 0.107143 | 0.107143 |
| external | music | 588 | 0.129252 | 0.119048 | 0.119048 | 0.125850 | 0.125850 | 0.122449 |
| song | file | 376 | 0.153706 | 0.127907 | 0.130814 | 0.130814 | 0.127907 | 0.132267 |
| song | voice | 376 | 0.185996 | 0.185996 | 0.184012 | 0.185996 | 0.185996 | 0.185996 |
| song | music | 376 | 0.145993 | 0.152010 | 0.152010 | 0.152010 | 0.152010 | 0.152010 |
| val | file | 600 | 0.109991 | 0.120073 | 0.073327 | 0.103270 | 0.101741 | 0.103270 |
| val | voice | 432 | 0.101836 | 0.150435 | 0.062512 | 0.134300 | 0.150435 | 0.101836 |
| val | music | 518 | 0.061750 | 0.042500 | 0.042500 | 0.052125 | 0.048127 | 0.056123 |
| test | file | 600 | 0.126644 | 0.136704 | 0.086665 | 0.123291 | 0.119938 | 0.119938 |
| test | voice | 414 | 0.142571 | 0.195609 | 0.091816 | 0.178785 | 0.195609 | 0.142571 |
| test | music | 535 | 0.046626 | 0.037301 | 0.037301 | 0.046626 | 0.046626 | 0.046626 |

### Slot 3 recommendation and verification

Recommend **sub102r3**, not the aggressive variant. Its declared objective is 0.124794788 versus sub102r3x 0.131537498 (0.674271 pp better). Pooled FILE/VOICE favor sub102r3 (0.114735/0.142593 versus 0.124105/0.153451); MUSIC is identical. The aggressive variant's much stronger mixture holdout does not outweigh this panel evidence, given the new official transfer observation.
sub102r3 remains only a narrow panel-objective improvement over officially winning sub102, with weaker pooled VOICE and MUSIC but stronger external MUSIC. Both r3 variants regress song MUSIC by ~1.40 pp versus sub100. The data have been reused for selection, and the separate-weight grid has more flexibility; no leaderboard improvement is promised.
An independent scalar implementation composed the canonical evaluator functions for all 3,956 files and checked all 60 family/group/head EERs against the vectorized search (tolerance 1e-12). Both retained single-model reference curves were reproduced exactly. All finite/range checks passed. The prior r3 ZIPs and their acceptance receipts were not changed; their report digests bind the report as it existed at release, before this append.


## r5 / sub107

Completed 2026-09-28T09:32:50 KST. No upload performed.

team-reported official context: sub102r3 0.8452467831 (ADS 0.8292380952) versus sub102 0.8418682116. r5 tests continued mixture training from the exact r3 epoch12 fp32 checkpoint, with fresh optimizer/seed 502, freeze-layers=0, 12 x 2500 steps, batch 8, backbone LR 1e-5/head LR 5e-4, 300-step cosine warmup, L2-SP 5 to the original pretrained weights, bf16 anchors/autocast. The convolutional feature extractor remains frozen, as in r3. Data is unchanged: v2 x2 + v3 train-only + 20% safe views. No early stopping was enabled; deadline 20:00 KST and 500-step request/checkpoint handling were retained.

Training finished with best validation record `{"epoch": 12, "score": 0.10109734612573873}`. Initial validation exactly reproduced all ten r3 epoch12 metrics. Exported fp16 epochs 8/10/12 were evaluated with the unchanged canonical scorer/evaluator on 600 v2 val + 600 test and 2756 panels. Every checkpoint has 3956/3956 coverage and zero scoring errors. All deploy pooling/affine fits use v2 val; no competition-test tuning.

Same pooled FILE/VOICE, .6 external + .4 pooled MUSIC objective; val-fit pooling; test tie-break. Best r5 epoch among 8/10/12.

| checkpoint | objective | F/V/M weights | standalone test FILE | VOICE | MUSIC |
|---|---:|---|---:|---:|---:|
| r3e12 | 0.124794788 | 0.150/0.100/0.350 | 0.104978 | 0.096663 | 0.078479 |
| r5e08 | 0.123026073 | 0.300/0.150/0.350 | 0.096724 | 0.103793 | 0.087804 |
| r5e10 | 0.127952882 | 0.200/0.050/0.250 | 0.101625 | 0.086969 | 0.091680 |
| r5e12 | 0.128009537 | 0.150/0.150/0.150 | 0.093371 | 0.086969 | 0.087804 |

Selected r5 checkpoint: **r5e08**; objective difference from sub102r3: **-0.176871 pp** (negative is better). The team authorized packaging the best r5 checkpoint; this comparison is evidence, not a claim of leaderboard improvement.
Panel weights `{"file": 0.3, "music": 0.35, "voice": 0.15}`; holdout-best `{"file": 0.85, "music": 0.4, "voice": 0.5}`; exact midpoint `{"file": 0.575, "music": 0.375, "voice": 0.325}`; constrained x weights `{"file": null, "music": 0.375, "voice": 0.3}`.
Exact midpoint to unsmoothed holdout-test best (smaller weight on ties); if a panel exceeds sub102r3 +1.5pp, retreat toward panel-best on a fixed <=.025 path; no accuracy re-optimization.
Constraint adjustments: `{"file": {"attempted": 12, "midpoint": 0.575, "selected": null}, "music": {"attempted": 1, "midpoint": 0.375, "selected": 0.375}, "voice": {"attempted": 2, "midpoint": 0.325, "selected": 0.3}}`.

| group | head | sub102r3 | sub107 | sub107x |
|---|---|---:|---:|---:|
| external | file | 0.166667 | 0.166667 | not built |
| external | music | 0.119048 | 0.125850 | not built |
| external | voice | 0.107143 | 0.103175 | not built |
| korean_mix | file | 0.157952 | 0.151676 | not built |
| korean_mix | music | 0.093750 | 0.093750 | not built |
| korean_mix | voice | 0.154684 | 0.151551 | not built |
| korean_speech | file | 0.050391 | 0.068555 | not built |
| korean_speech | voice | 0.048633 | 0.045117 | not built |
| pooled | file | 0.114735 | 0.108946 | not built |
| pooled | music | 0.114070 | 0.113493 | not built |
| pooled | voice | 0.142593 | 0.139225 | not built |
| song | file | 0.127907 | 0.125000 | not built |
| song | music | 0.152010 | 0.153994 | not built |
| song | voice | 0.185996 | 0.185996 | not built |
| test | file | 0.136704 | 0.096724 | not built |
| test | music | 0.037301 | 0.046626 | not built |
| test | voice | 0.195609 | 0.149701 | not built |
| val | file | 0.120073 | 0.080048 | not built |
| val | music | 0.042500 | 0.048127 | not built |
| val | voice | 0.150435 | 0.125024 | not built |

### sub107

ZIP `$PROJECT_DIR\submissions\sub107.zip`, 8,697,570,699 bytes; SHA256 `6192bdb1a9b77e627473365e8f990e63ca93bd04c2c6d5aaba37817de012ade8`.
Runner `70626da8716795bc837ea335bea464175eb8e28fb6752e82f6061ebb029877cd`; overlay `e2494e8549255942452547832bad0a118fa242e6fe11d21b5d9c194f1cde0b88`; source revision `3310b60892e83d210b308b83b1e3d489e14b0145`. Exact config/generator/build-tool digests bind the uncommitted source.
All five CPU tests, archive validation, 24-file exact GPU parity, normal/reversed offline smoke, exact order independence, unchanged presences, neutral corrupt fallback, zero healthy warnings/network attempts and post-smoke hashes passed.
Smoke normal/reversed seconds: 84.10/70.81. Added-pass mean 0.03861s/file, p95 0.05164s; projected runtime 40.80 min, conservative 48.96 min (<60).

### r5 risks and tooling

Panels and local mixture holdouts are reused development sets; holdout-best weights can substantially overstate official transfer. The official r3 gain motivates the continuation but does not guarantee further improvement. Runtime is a shared-GPU added-pass projection anchored to the prior ~40-minute sub100 run, not a fresh official L4 timing.
Final stack preparation: `scratchpad/final_stack_tooling.md`, `tools/make_final_stack_runner.py`, `tools/build_final_stack_package.py`, `tools/run_final_stack_release.py`, and `tests/test_final_stack.py`. CPU composition/ownership tests passed. Actual combined-winner GPU parity/runtime is deferred until team supplies final components/weights; no unsupported combined accuracy or <60-minute timing claim is made.
Training provenance: `runs/sub102-nii-ml/r5/provenance.json`; full comparisons/selections: `sub107_epoch_comparison.json`, `sub107_selection.json`. Existing r2/r3/sub104/sub106 data and submitted ZIPs remain unchanged.

### sub107x constrained midpoint follow-up

The automatic r5 release stopped the midpoint retreat at panel-best FILE .30. That point exceeds the Korean-speech sub102r3 bound by 1.8164 pp, so the original table above says `sub107x` was not built. A fixed-grid feasibility audit found FILE .25 as the closest admissible weight to the holdout midpoint .575. This follow-up supersedes that provisional omission; no new accuracy optimization or competition-test tuning was used.

Weights FILE/VOICE/MUSIC: **0.275/0.300/0.375**. Original midpoint: {'file': 0.575, 'music': 0.375, 'voice': 0.325}. Selection objective 0.125725834. Every labeled panel is within +1.5 pp of released sub102r3.

| panel | head | sub102r3 EER | sub107x EER | gap (pp) |
|---|---|---:|---:|---:|
| korean_speech | file | 0.050391 | 0.060742 | +1.035 |
| korean_speech | voice | 0.048633 | 0.058203 | +0.957 |
| korean_mix | file | 0.157952 | 0.151676 | -0.628 |
| korean_mix | music | 0.093750 | 0.093750 | +0.000 |
| korean_mix | voice | 0.154684 | 0.157816 | +0.313 |
| external | file | 0.166667 | 0.161310 | -0.536 |
| external | music | 0.119048 | 0.125850 | +0.680 |
| external | voice | 0.107143 | 0.103175 | -0.397 |
| song | file | 0.127907 | 0.123547 | -0.436 |
| song | music | 0.152010 | 0.153994 | +0.198 |
| song | voice | 0.185996 | 0.176011 | -0.998 |

ZIP `submissions/sub107x.zip`, 8,697,570,697 bytes; SHA256 `0d2c13d02d70ba824d64665c6c75ba661993588015ae857191bc4fdb9ec85b8f`. Runner SHA256 `0a97693b05d9637237d77817085f027e305e447c9f3ef2beda8512f68cb1e42d`; overlay SHA256 `e2494e8549255942452547832bad0a118fa242e6fe11d21b5d9c194f1cde0b88`.
Five CPU tests, archive validation, 24-file exact GPU runner/scorer parity, normal and reversed offline smoke, exact order independence, unchanged presences, corrupt fallback, zero healthy warnings/network, and post-smoke ZIP/member hashes passed.
Added-pass mean 0.03690s/file; projected official runtime 40.77 min and conservative 48.92 min (<60).
Selection/guard receipt: `sub107x_selection.json`, separately bound to `sub107x_config.json`. Original `sub107_selection.json` and `sub107` package remain immutable. No upload performed.


## r6 / sub108

Completed 2026-09-28T13:38:23.292455 KST. No upload performed.


v4: 30,000 new train-only renders, all five layouts, four label triples and 15 channel conditions. SONICS was on hold when v4 was prepared and is absent from that corpus; the owner approved its training use later on 2026-09-28 under CC BY-NC 4.0; the fake-vocal music recipes were omitted because that pool contained only SONICS, while all label triples remain represented. Exact source hashes, panel blocklist, v2/v3 val/test group+hash exclusions, output hashes, component spans and codec delays passed the independent audit. See mixture_v4/SUMMARY.md for every used-source licence.

Initial r6 validation exactly reproduced r5 epoch08. Training best validation: `{"epoch": 3, "score": 0.10154654764075631}`. Exported fp16 epochs 8/10/12 used the canonical scorer on 600 v2 val, 600 v2 test and all 2,756 panel files. All selection fits use v2 val; no competition-test data used.

Same pooled FILE/VOICE, .6 external + .4 pooled MUSIC objective; val-fit pooling; test tie-break. Best r6 epoch among 8/10/12.

| checkpoint | objective | F/V/M weights | standalone test FILE | VOICE | MUSIC |
|---|---:|---|---:|---:|---:|
| r3e12 | 0.124794788 | 0.150/0.100/0.350 | 0.104978 | 0.096663 | 0.078479 |
| r5e08 | 0.123026073 | 0.300/0.150/0.350 | 0.096724 | 0.103793 | 0.087804 |
| r6e08 | 0.125198843 | 0.200/0.100/0.300 | 0.111684 | 0.108641 | 0.104882 |
| r6e10 | 0.127430838 | 0.150/0.150/0.300 | 0.101625 | 0.113488 | 0.091680 |
| r6e12 | 0.126404926 | 0.200/0.200/0.350 | 0.090018 | 0.113488 | 0.087804 |

Selected **r6e08**; objective vs r5e08 +0.21728 pp and vs r3e12 +0.04041 pp (negative is better).
Panel weights `{"file": 0.2, "music": 0.3, "voice": 0.1}`. Holdout-best `{"file": 0.65, "music": 0.3, "voice": 0.5}`. Constrained x `{"file": null, "music": 0.3, "voice": 0.27499999999999997}`. Exact midpoint to unsmoothed holdout-test best (smaller weight on ties); if a panel exceeds sub102r3 +1.5pp, retreat toward panel-best on a fixed <=.025 path; no accuracy re-optimization.

| group | head | sub102r3 | sub108 | sub108x |
|---|---|---:|---:|---:|
| external | file | 0.166667 | 0.183929 | not built |
| external | music | 0.119048 | 0.122449 | not built |
| external | voice | 0.107143 | 0.123016 | not built |
| korean_mix | file | 0.157952 | 0.163191 | not built |
| korean_mix | music | 0.093750 | 0.090625 | not built |
| korean_mix | voice | 0.154684 | 0.148418 | not built |
| korean_speech | file | 0.050391 | 0.054688 | not built |
| korean_speech | voice | 0.048633 | 0.046875 | not built |
| pooled | file | 0.114735 | 0.120525 | not built |
| pooled | music | 0.114070 | 0.109728 | not built |
| pooled | voice | 0.142593 | 0.137711 | not built |
| song | file | 0.127907 | 0.125000 | not built |
| song | music | 0.152010 | 0.145993 | not built |
| song | voice | 0.185996 | 0.185996 | not built |
| test | file | 0.136704 | 0.115037 | not built |
| test | music | 0.037301 | 0.031852 | not built |
| test | voice | 0.195609 | 0.190762 | not built |
| val | file | 0.120073 | 0.095020 | not built |
| val | music | 0.042500 | 0.056123 | not built |
| val | voice | 0.150435 | 0.155072 | not built |

### sub108

ZIP `submissions/sub108.zip`, 8,697,568,012 bytes; SHA256 `1e400fa2bd6c173acd857c3d642edd044478e10384d27230ccb2ffaead737870`. Runner `b962b0632537055000e1edf9d30f48f0d2fff33d78b8c5cba71e3a45e6bac943`, overlay `6e3180a968bf21f8eec06fc5f360a7f6e7aefe457761b317f0a1073609fcf896`, source revision `3310b60892e83d210b308b83b1e3d489e14b0145`.
All five CPU tests, archive validation, 24-file exact runner/scorer parity, normal and reversed GPU smoke, exact order independence, unchanged presences, neutral corrupt fallback, zero healthy warnings/network attempts and post-smoke hashes passed.
Smoke normal/reversed 166.54/73.01s. Added pass 0.04552s/file mean, 0.05915s p95; projected runtime 40.93 min, conservative 49.12 min (<60).



## sub109 ensemble

Completed 2026-09-28T13:46:19 KST. **NO-GO** under the predeclared +0.10 pp margin; no package, GPU job or upload was created.

Retained fp16 window scores from r5e08 (v2+v3 training) and r6e08 (v2+v3+v4) cover the same 600 v2 val, 600 v2 test, and 2,756 panel files. All keys, labels, presence masks, window starts, counts and paths agree. Each model keeps its own val-fitted pooling and affine calibration. The third r3e12 score set is used only for an information-only three-model mean; no three-model ZIP was considered because of the 10 GB limit. No competition-test data were used for tuning.

Mean of r5e08 and r6e08 val-pooled per-head probabilities; one 0.05-grid logit blend per head into sub100; FILE pre-floor then original floor; FILE/VOICE pooled EER, MUSIC .6 external+.4 pooled, test tie-break.

| candidate | FILE objective | VOICE objective | MUSIC objective | mean objective | F/V/M weights |
|---|---:|---:|---:|---:|---|
| sub107 r5e08 | 0.108946 | 0.139225 | 0.120907 | 0.123026073 | 0.30/0.15/0.35 |
| r5e08+r6e08 | 0.116105 | 0.139225 | 0.117863 | 0.124397780 | 0.30/0.10/0.35 |
| r3e12+r5e08+r6e08 (information only) | 0.113630 | 0.139983 | 0.118899 | 0.124170859 | 0.25/0.10/0.35 |

The requested two-model mean is **0.124397780**, **0.137171 pp worse** than sub107 (0.123026073); it exceeds the +0.10 pp GO boundary by **0.037171 pp**. The information-only three-model mean is 0.124170859. A conservative two-model ZIP size preflight was 9,321,085,685 bytes, below 10 GB; size was not the NO-GO reason.

### Per-panel and local holdout EERs

| group | head | n | sub107 | r5+r6 | r3+r5+r6 info |
|---|---|---:|---:|---:|---:|
| pooled | file | 2756 | 0.108946 | 0.116105 | 0.113630 |
| pooled | music | 1604 | 0.113493 | 0.110983 | 0.108473 |
| pooled | voice | 2672 | 0.139225 | 0.139225 | 0.139983 |
| korean_speech | file | 1152 | 0.068555 | 0.066016 | 0.054688 |
| korean_speech | voice | 1152 | 0.045117 | 0.048633 | 0.051172 |
| korean_mix | file | 640 | 0.151676 | 0.157952 | 0.157952 |
| korean_mix | music | 640 | 0.093750 | 0.090625 | 0.087500 |
| korean_mix | voice | 640 | 0.151551 | 0.151551 | 0.151551 |
| external | file | 588 | 0.166667 | 0.179762 | 0.173214 |
| external | music | 588 | 0.125850 | 0.122449 | 0.125850 |
| external | voice | 504 | 0.103175 | 0.119048 | 0.115079 |
| song | file | 376 | 0.125000 | 0.100654 | 0.125000 |
| song | music | 376 | 0.153994 | 0.152010 | 0.153994 |
| song | voice | 376 | 0.185996 | 0.192012 | 0.193996 |
| val | file | 600 | 0.080048 | 0.073327 | 0.088299 |
| val | music | 518 | 0.048127 | 0.048127 | 0.046498 |
| val | voice | 432 | 0.125024 | 0.155072 | 0.150435 |
| test | file | 600 | 0.096724 | 0.090018 | 0.108331 |
| test | music | 535 | 0.046626 | 0.037301 | 0.037301 |
| test | voice | 414 | 0.149701 | 0.190762 | 0.195609 |

The predeclared +0.10 pp tolerance acknowledges that local panels under-predicted r3's official gain, but the r5+r6 mean falls outside it. Reused panels and mixture holdouts are selection-biased; this NO-GO does not prove poor leaderboard transfer. Exact single-model sub107 curves were reproduced. An independent scalar implementation checked all 40 ensemble/group/head EERs against the vectorized curves to 1e-12. Evidence: `sub109_selection.json`, `sub109_curves.json`, and `runs/sub109-ensemble/sub109_evaluation.validation.json`. Existing ZIPs and other researchers' files were untouched.


## r7 / sub110

Completed 2026-09-28T18:04:33.235937 KST. **GO**. No upload performed.


The frozen manifest has 6341 A/B-union real files, 222 newly allowed SONICS-dependent rows and 30000 v4 rows. Two SONICS panel rows with no-derivatives-licensed other parents were excluded. Training items SHA256 `1b3413968b301370fe01eb687222d6b7783a51ae9a671be68b4fae89bd0b33e6`. No honest held-out panel remains after all-data training; panels were not scored. Epoch 5 was fixed in advance, without validation selection.

Warm-start validation exactly reproduced r5 e08. Final training validation mean3_lse1 0.098974156; this did not select the checkpoint. The canonical scorer covered all 1,200 mixture_v2 holdout files, and canonical pooling/affine calibration used only the 600-file v2 val split. Published blend weights FILE/VOICE/MUSIC were fixed at 0.20/0.40/0.30 from sub106 OOF development.

Each head on the combined 1,200-file v2 val+test holdout at fixed sub110 weights must be <= sub107's own selected weights +1 pp; val/test separately are diagnostic.

| split | head | n | sub107 EER | sub110 EER | delta pp |
|---|---|---:|---:|---:|---:|
| val | file | 600 | 0.080048 | 0.088299 | +0.8250 |
| val | voice | 432 | 0.125024 | 0.053237 | -7.1787 |
| val | music | 518 | 0.048127 | 0.052125 | +0.3998 |
| test | file | 600 | 0.096724 | 0.108331 | +1.1607 |
| test | voice | 414 | 0.149701 | 0.082122 | -6.7580 |
| test | music | 535 | 0.046626 | 0.041178 | -0.5449 |
| combined | file | 1200 | 0.090839 | 0.098322 | +0.7483 |
| combined | voice | 846 | 0.139472 | 0.064996 | -7.4476 |
| combined | music | 1053 | 0.049333 | 0.044596 | -0.4736 |

### Package

ZIP `submissions/sub110.zip`, 8,697,573,572 bytes, SHA256 `ecc4bfcf411e29f72dc2a862192ec084be98a7bcc4178dc2cc44655f30956805`. Overlay SHA256 `bc02b4b2200d4bc3fe257ca5d043f1ad7eba73a8d3c74fb1954898d9b32d56a4`; runner SHA256 `b9c0aab86fe8b9a4417201e34521a39657181a17233f43dc2f376ae4ef1e075b`; source revision `3310b60892e83d210b308b83b1e3d489e14b0145`.
Five CPU tests, archive validation, 24-file exact runner/scorer parity, normal/reversed offline GPU smokes, unchanged presence and corrupt fallback, exact order independence, zero healthy warnings/network attempts, member and ZIP post-smoke hashes passed. Added pass mean 0.04260s/file, p95 0.05806s; projected runtime 40.88 minutes, conservative 49.05 minutes (<60).

The mixture holdout has been used repeatedly for development; its safety check is not an official-score prediction. This all-data continuation has inherited exposure through r5 initialization, and the panel data were training inputs. No competition-test data or upload were used.
