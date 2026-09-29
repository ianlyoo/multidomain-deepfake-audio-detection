# Curated source report

Historical research report retained for technical details. Local paths are portable placeholders; internal operations lines were omitted. Later official results and final selection are documented in `REPORT.md` and `docs/experiments.md`.

NO-GO — sub97: the FT-MUSIC segment head retrained on runner-matched window bags improves ranking but fails the pre-declared real-music FP checks on all three music panels.

2026-09-27 KST, Opus researcher. Plan: `sub97_plan.json` (sha256 b840bd59…8a4b), declared at 04:34:30 KST. The fit ran at 04:41:10 and the panel evaluation at 04:41:25, so the plan predates every candidate head and every panel metric. There was no build, no upload and no commit.

## What was tested

- sub97 = sub90 with only the segment path's FT head replaced.
- MUSIC = logit_blend(MUSIC17, FT_seg(new head), .5).
- Everything else is kept from sub90: the windows, the PANNs gate (.20 / .80), the fallback, round10, VOICE, the presences, and the whole-file release-v2 FT path that feeds FILE and the floor.

**Features.** Each training file becomes one bag: the presence-weighted mean of its kept 4 s window fakeprints.
- Extraction used the sub90 runner's own `load_audio`, `predict_presence` (PANNs Cnn14), `load_lofcz_audio`, `make_lofcz_fakeprint` and `_sub80_window_starts`. They were imported from the digest-checked `script_sub90.py`, with the model files from the sub90 smoke projection.
- The head is linear, so head(bag) equals the runner's presence-weighted mean window logit.
- Panel parity: release-v2 on the panel bags reproduces the cached sub80 `segment_ft` to 2.0e-14 on all 1592 gated files. After the blend it reproduces published sub90 MUSIC exactly (0 differing rows) on all four panels.

**Training data.** The manifest is `sub97_train_manifest.jsonl`, sha256 603c67d157a33202c865aa2d0ddfe3ec5175ff5831dcf35da5cd0936cd8ee9d0. **Erratum:** `sub97_plan.json` quotes a stale prefix, f2ca92ed, from the pre-fix manifest run. The correct digest is recorded in `sub97_plan_erratum.json` and `sub97b_plan.json`. The plan file stays byte-identical (sha b840bd59…), and the manifest (04:31:04) predates it (04:34:30).
- **release-v2 Echoes/FMA parents** from views-v1, fma-extra and fma-heldout-release.
  - Deduplicated by sha256 (1501 duplicates dropped).
  - sub54-clean excluded.
  - Scored as the original clean files. The release-v2 overlay pool is the korean_speech panel's source, so its renders were not reused.
- **mixture_v2 slice:** music-present sequential/simultaneous/overlap/partial examples, or single-modality examples on non-clean channels. Random cap 3000. It is already AI-Hub-free and panel-source-free.
- **Panel blocklist:** every korean_mix, external and song music content id, track, sha256 and path, plus the korean_speech audio. It removed **1090 release-v2 rows**: korean_mix and external are built on the same paired_music FMA/Echoes contents.
- **Groups:** union-find over music-source tokens (FMA track and artist, Echoes content, SONICS/MusicCaps ids). That gives 2443 components in 5 folds. Speech ids are excluded from grouping; with them, everything chained into one 4086-row component.
- **Bags:** 7456 files → 7255 bags. 199 files were ungated (the runner would fall back) and 2 failed to decode (NoBackendError).
  - release-v2 parents: 909 fake / 3481 real.
  - mixture_v2: 1158 fake / 1707 real.
- **Extraction cost:** 2 shards × ~500 s, peak 2.86 GiB VRAM each, 3 GB leases (released), RSS ≤ 2 GB.

**Arms.**
- **A_anchored (decision arm):** weighted BCE + λ/2‖w − w_release_v2‖², bias free, started at release-v2.
  - Weights: in each class, the release-v2 block gets 2/3 and mixture_v2 1/3, with fake families equal.
  - λ = 0.01 was chosen by the 5-fold source-group holdout. 0.003 was 0.04 pp lower, so the tie rule picked the larger λ.
  - ‖Δw‖ = 2.59, **bias shift +6.49**.
- **B_affine (diagnostic only):** a·z + c recalibration of release-v2 (a .886, c +3.21).

## Gate (arm A; sub90 thresholds frozen per panel)

| Check | sub90 | sub97 A | Result |
|---|---:|---:|---|
| Source-group holdout pooled EER (release-v2 vs A) | 11.36 % | 9.33 % | PASS |
| external all/all MUSIC EER | 16.667 | 14.286 (−2.38 pp) | PASS |
| external strict/all | 18.095 | 15.714 (−2.38 pp) | PASS |
| telephone pooled (kmix+ext+song, n=292) | 19.52 | 18.15 (−1.37 pp) | PASS |
| song all/all | 15.399 | 13.800 (−1.60 pp) | PASS |
| korean_mix all/all (corroboration) | 9.375 | 9.375 (0) | PASS (flat, no corroborating gain) |
| real-music FP, korean_mix | 30 | **147** | **FAIL** |
| real-music FP, external | 49 | **176** | **FAIL** |
| real-music FP, song | 19 | **42** | **FAIL** |

The holdout breaks down as follows:
- By source: mixture_v2 22.6 → 17.5 %; release-v2 parents 3.18 → 3.53 %. release-v2 had trained on those parents, so its baseline there is optimistic.
- By channel: clean 8.6 → 7.1, mp3 19.3 → 14.6, telephone 25.5 → 21.1.

Arm B (diagnostic) also explodes FP: 190 / 223 / 54. Its EER is mostly flat, and it worsens song telephone.

## Why it fails

- The retrained head is calibrated to balanced classes on window bags. On the training bags its mean logits are real −4.3 / fake +5.4 on release-v2 parents, versus −7.5 / +1.5 for release-v2.
- In the .5 blend, a +6.5 bias moves every gated file's MUSIC logit up by about 3.2. Ranking among gated files is unchanged, but everything crosses the frozen sub90 threshold: FN drop to single digits and FP multiply.
- EER is threshold-free, and its gains come from the weight change: external −2.4 pp on every channel, song −1.6, telephone −1.4.
- korean_mix, the panel that predicted sub80, doesn't move at all overall. It loses 2 crossing steps on mp3_64 (2.71 → 8.13), the same slice that hurt sub92 and sub95.

## Post-hoc diagnostic (not a decision arm; written after the NO-GO)

`sub97_posthoc_recenter.py` / `.json` tests arm A's weights with the bias recentred by a training-only rule: A's mean logit on training real bags is set equal to release-v2's (Δ = −3.08).
- EER is unchanged on every slice: the bias only affects gated vs. ungated files.
- Telephone pooled improves to 17.47 (−2.06 pp).
- FP at the frozen thresholds: korean_mix 30 → 32, external 49 → 36, song 19 → 20.
- FN: 30 → 30, 49 → 45, 39 → 31.

Even this calibration-fixed head fails the declared FP checks, narrowly: korean_mix +2 and song +1.

## Interpretation and risks

- The contrary evidence stands. korean_mix, the only panel that predicted the segment-scoring win, shows no gain, and its mp3_64 slice degrades.
- The external panel overstated sub84's new-head gain (−3.3 pp predicted vs about −0.3 official). This is also a newly trained head, so its −2.4 pp should be discounted the same way.
- Direction is still consistent across external, song and telephone, and the holdout gain is on data the old head never saw. Window-matched training does carry ranking information.
- If the team wants to pursue it, the next step is a new pre-declared plan (e.g. sub97b = recentred bias, the rule above). The owner's standing gate rule would make it an explicit exploratory decision for the owner, because it fails the FP criterion as declared. I would not upload it under the current rule.

## Files (all in $DATA_DIR/derived/mixture_v2)

- **Scripts:** `sub97_common.py`, `sub97_panel_bags.py`, `sub97_manifest.py`, `sub97_train_bags.py`, `sub97_fit.py`, `sub97_eval.py`, `sub97_posthoc_recenter.py`
- **Plan:** `sub97_plan.json` (+ `.sha256`)
- **Manifest:** `sub97_train_manifest.jsonl` / `.json`
- **Bags:** `sub97_train_bags/` (201 MB, reconstructible) and `sub97_panel_bags.npz` (44 MB, reconstructible)
- **Head:** `sub97_music_ft_seg.json` (sha aa4de745…edb6; passes the runner validator; not packaged)
- **Fit outputs:** `sub97_fit_summary.json`, `sub97_holdout_oof.npz`
- **Panel outputs:** `sub97_music_metrics.csv`, `sub97_music_predictions.csv`, `sub97_music_summary.json`
- **Logs:** `sub97_*.log`

Storage: about 245 MB of new derived data, all reconstructible from the scripts in this directory. It can be deleted if sub97b is not pursued.

## sub97b (exploratory, post-hoc): recentred-bias segment head — built, validated, smoked

Status: **EXPLORATORY, not gate-passing, no upload authority.** Added 2026-09-27 ~05:00 KST at the team's request, pending a team decision.

**Honesty note.** The recentring rule was chosen after the sub97 NO-GO had shown that arm A failed the FP checks through its +6.49 bias shift. The rule uses training data only and has no panel-tuned quantity, but its choice was motivated by the panel results. It also fails the declared sub97 FP checks: korean_mix 30→32, song 19→20. The plan is `sub97b_plan.json` (sha256 72cbd2f5…f099); it records this and carries the full metrics table.

**Head.** `sub97b_music_ft_seg.json`, sha256 7fcc13317c6b776d74b5af31278254027654510338dd4c5a4999f205033a9e10.
- Weights: sub97 arm A.
- Bias: bias_A − mean over the 5188 training real-music bags of (z_A − z_release-v2) = 7.9041 − 3.0821 = **4.8220**. release-v2's bias is 1.4145.
- Mean training logits (real / fake):
  - release-v2: −6.92 / +0.10
  - sub97b: −6.92 / +1.29
  - arm A: −3.84 / +4.38
- Receipt: `sub97b_head_receipt.json`.

**Panel metrics vs sub90** (`sub97b_eval.py` → `sub97b_music_metrics.csv`; FP/FN at the frozen sub90 crossing)

| Panel | Subset | Channel | N | sub90 EER % | sub97b EER % | Δ pp | FP 90→97b | FN 90→97b |
|---|---|---|---:|---:|---:|---:|---|---|
| korean_mix | all | all | 640 | 9.375 | 9.375 | 0.000 | 30→32 | 30→30 |
| korean_mix | all | clean | 492 | 8.943 | 8.943 | 0.000 | 19→22 | 25→22 |
| korean_mix | all | telephone | 74 | 17.546 | 17.546 | 0.000 | 9→6 | 4→7 |
| korean_mix | all | mp3_64 | 74 | 2.711 | 8.132 | **+5.421** | 2→4 | 1→1 |
| external | all | all | 588 | 16.667 | 14.286 | −2.381 | 49→36 | 49→45 |
| external | all | clean | 196 | 12.245 | 10.204 | −2.041 | 11→10 | 12→8 |
| external | all | telephone | 196 | 21.429 | 18.367 | −3.061 | 23→11 | 17→23 |
| external | all | mp3_64 | 196 | 17.347 | 14.286 | −3.061 | 15→15 | 20→14 |
| external | strict | all | 420 | 18.095 | 15.714 | −2.381 | 34→25 | 39→36 |
| external | strict | clean | 140 | 14.286 | 8.571 | −5.714 | 6→6 | 10→6 |
| external | strict | telephone | 140 | 21.429 | 18.571 | −2.857 | 17→8 | 15→19 |
| external | strict | mp3_64 | 140 | 18.571 | 15.714 | −2.857 | 11→11 | 14→11 |
| song | all | all | 376 | 15.399 | 13.799 | −1.600 | 19→20 | 39→31 |
| song | all | clean | 333 | 14.990 | 14.095 | −0.895 | 17→19 | 35→27 |
| song | all | telephone | 22 | 14.583 | 14.583 | 0.000 | 1→0 | 2→2 |
| song | all | mp3_64 | 21 | 21.324 | 5.882 | −15.441 | 1→1 | 2→2 |
| song | strict | all | 241 | 20.078 | 16.622 | −3.456 | 14→16 | 20→16 |
| song | strict | clean | 202 | 21.767 | 19.077 | −2.690 | 12→15 | 18→14 |
| song | strict | telephone | 20 | 6.250 | 6.250 | 0.000 | 1→0 | 2→2 |
| song | strict | mp3_64 | 19 | 3.333 | 0.000 | −3.333 | 1→1 | 0→0 |
| telephone pooled (kmix+ext+song) | | | 292 | 19.522 | 17.466 | −2.056 | | |

Real-music FP per panel (all/all): korean_mix 30→32, external 49→36, song 19→20. The source-group holdout (same weights) is 11.36 → 9.33 %.

**Package.** `submissions/sub97b.zip`:
- SHA256 **e70a944ab2788549bc351152e5ed6a2f0c8e3ceedae447c10c993af25efd0bcd**, 8,120,877,371 B, 57 members.
- Memo string: `sub97b sha256:e70a944ab2788549`.
- Runner `submission/script_sub97b.py`, sha256 83910bc2b01e17373eae6c4aef7c5c8d1c808ab14067177637609fb1bbccb29e.
  - Generated from sub90 (3f0279e9…) by `tools/make_sub97b_runner.py`; the build tool re-derives it and requires byte equality.
  - Diff: `submissions/sub97b.runner.diff`. The only non-additive edit is the model argument of the `_sub80_final_music` call.
- Member diff vs sub90: added `model/ft_music_seg_sub97b.json`; changed script.py, MODEL_INFO.txt, SHA256SUMS.txt; 53 unchanged.
- The whole-file FT (`ft_music_fakeprint_v2.json`, release-v2 e9391e80…) still feeds FILE and the floor. A segment-head load failure keeps release-v2, i.e. exact sub90 MUSIC.

**Validation.**
- **CPU tests:** `tests/test_sub97b_music.py`, 4/4 passed:
  - the generator reproduces the runner;
  - only the segment call line changes;
  - the loader pins the digest;
  - replay: running the runner's own `_sub80_final_music` on real panel audio with the cached PANNs reproduces panel sub97b MUSIC **exactly** (max diff 0.0) on all 1592 gated files, and sub90 published MUSIC exactly with the release-v2 head. Ungated rows equal sub90.
- **Archive:** `tools/validate_submission.py` passed (57 files, 8,839,813,094 B extracted).
- **Smoke projection:** `$DATA_DIR/package-smoke/sub97b-v1`, hash-verified.
- **GPU smoke:**
  - Normal and reversed both passed, with 0 network attempts, reversed == normal, and a neutral corrupt row. There are no WARNING lines except TEST_CORRUPT.
  - vs sub90, only MUSIC differs (14/22 rows); FILE, VOICE and both presences are byte-identical.
  - Minimum available RAM was 6.25 GiB; no OOM.
- **Receipts:** `submissions/sub97b.build.json` (acceptance complete), `sub97b.validation.json`, `sub97b.resources.json`, `sub97b.archive_validation.json`. Finalized by `scratchpad/finalize_sub97b.py`.

**Runtime (smoke child_wall_seconds vs the sub90 smoke):**

| Run | sub90 | sub97b | Ratio |
|---|---:|---:|---:|
| Normal | 41.08 s | 47.26 s | 1.150 |
| Reversed | 39.99 s | 39.66 s | 0.992 |
| Pooled | | | 1.072 |

The runner does identical work: the same windows and fakeprints, plus one extra 78 KB JSON load and digest. The normal-run gap is noise from the cold first run, with video running concurrently. Expected official runtime equals sub90's.

**Risks.**
- korean_mix, the panel that predicted sub80, shows no gain, and its mp3_64 slice loses 2 crossing steps.
- The external panel overstated sub84's new-head gain.
- The post-hoc choice of the recentring rule.
- If uploaded, it is a one-slot test of whether window-matched head training transfers.
