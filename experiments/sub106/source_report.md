# Curated source report

Historical research report retained for technical details. Local paths are portable placeholders; internal operations lines were omitted. Later official results and final selection are documented in `REPORT.md` and `docs/experiments.md`.

# sub106 r4 real-domain fine-tuning

Updated 2026-09-28T14:14:19.307044 KST. No upload or ZIP modification.


## Split and provenance

2756 panel rows in 186 connected source groups. Connected by speaker, song/content, original source hashes/paths, and copied panel files; SONICS variants with a common content number share a group. No source token crosses A/B. Evaluation preserves the canonical panel row weighting, including historical copies; training deduplicates physical paths.

| Panel | Fold A | Fold B |
|---|---:|---:|
| korean_speech | 596 | 556 |
| korean_mix | 337 | 303 |
| external | 294 | 294 |
| song | 188 | 188 |

Fold A model trains A and evaluates B; fold B reverses this. The 517 non-SONICS mixture validation files alone fit the unchanged `sub102_eval.fit_pooling` tau/gamma/affine procedure. The remaining 83 SONICS-containing validation files are scoring-only. This calibration subset was fixed before either fold was scored and remains unchanged after licence approval; sub102r3 retains its published calibration for the exact incumbent comparison. Epoch 3 is the common checkpoint for both matched folds. Primary weights FILE/VOICE/MUSIC 0.15/0.10/0.35 match the incumbent and were fixed before r4 scoring. Weight curves are a separately labelled development analysis, without claiming unbiased accuracy at their optima.

Training additions (unique files before fold exclusions):

| Data | Files | Licence / attribution |
|---|---:|---|
| Korean panel speech/mixtures | fold counts above | Zeroth CC-BY-4.0; Chatterbox/BigVGAN MIT; MMS CC-BY-NC-4.0; XTTS CPML noncommercial; FMA/Echoes per-parent BY/SA/PD, retained manifests |
| External panel mixtures | 588 | ITW CC-BY-SA-4.0 plus retained FMA/Echoes parent licences |
| Original FMA song panel files admitted | 32 | Per-track CC-BY / public-domain; attribution retained |
| Extra ITW originals | 2279 | CC-BY-SA-4.0; Müller et al., In-the-Wild; original file/speaker/label retained |
| Extra Jamendo originals | 315 | Per-track CC-BY/NC/SA, derivatives allowed, no ND; artist, track URL and licence URL retained |
| Existing sub41 clean music | 145 | FMA BY/PD and Echoes CC-BY-SA-4.0, per-parent attribution retained |
| Separate Korean-extra historical probe | 1190 | FLEURS CC-BY-4.0; OmniVoice CC-BY-NC; Qwen Apache-2.0; Vocos MIT; Supertonic OpenRAIL-M. Same Zeroth speaker aliases kept fold-local |

The separate Korean-extra probe was previously a local probe and is now explicitly repurposed for r4 training; it is not independent validation. ITW/Jamendo/sub41 additions are outside these four panels; linked speakers/content are still purged from the opposing fold. Fifty-nine calibration-source overlaps were excluded from real additions. Exact training manifests, per-file licence/source metadata and source-manifest hashes are in `sub106_items_A/B.jsonl`, `sub106_real_provenance.jsonl`, and `sub106_split_audit.json`.

**SONICS provenance:** the owner approved SONICS audio on 2026-09-28 as CC BY-NC 4.0 for this non-commercial competition; `docs/provenance_register.csv` records the decision and remaining attribution/ToS notes. The earlier r2/r3 SONICS training provenance item is resolved. Per team, this already-fixed r4 configuration was not restarted: 224 SONICS-dependent panel rows and 4273 SONICS-containing mixture rows remain excluded from new r4 training; SONICS is evaluation-only in this run.

**Inherited exposure:** 708/816 r2 speech-view rows with declared opposing held-panel music-content group matches were removed from new A/B training. An independent audit confirmed actual parent-source matches in 624/714 of those rows; the remainder are conservative cohort exclusions. These matches were content, not speaker tokens. R2 itself was trained before these folds existed. These are held-out tests of the *new continuation*, not fully unseen-from-initialization source tests. R3 and the candidate r4 folds share the r3 e12 initialization; r3 adds further inherited mixture exposure. Panels also influenced earlier sub102 design/weights; neither these comparisons nor weight-selected estimates are pristine generalization estimates.

## Held-out panel EERs

EER fractions; lower is better. Primary numbers use fixed epoch and fixed weights. FILE replaces the NII fine-tune input at the sub100 pre-floor blend; its original floor remains unchanged. VOICE/MUSIC use the unchanged canonical sub102 blend procedure. Presence outputs remain unchanged.

| Train fold → held fold | Panel | Head | n | sub102r3 | r4 fixed | Δ pp |
|---|---|---|---:|---:|---:|---:|
| A → B | pooled | file | 1341 | 0.115668 | 0.113422 | -0.225 |
| A → B | pooled | voice | 1299 | 0.132387 | 0.128575 | -0.381 |
| A → B | pooled | music | 785 | 0.123571 | 0.121010 | -0.256 |
| A → B | korean_speech | file | 556 | 0.044988 | 0.048594 | +0.361 |
| A → B | korean_speech | voice | 556 | 0.044988 | 0.048594 | +0.361 |
| A → B | korean_mix | file | 303 | 0.129094 | 0.131307 | +0.221 |
| A → B | korean_mix | voice | 303 | 0.155154 | 0.148548 | -0.661 |
| A → B | korean_mix | music | 303 | 0.079209 | 0.079209 | +0.000 |
| A → B | external | file | 294 | 0.190476 | 0.179762 | -1.071 |
| A → B | external | voice | 252 | 0.095238 | 0.103175 | +0.794 |
| A → B | external | music | 294 | 0.156463 | 0.156463 | +0.000 |
| A → B | song | file | 188 | 0.147592 | 0.147592 | +0.000 |
| A → B | song | voice | 188 | 0.196786 | 0.196786 | +0.000 |
| A → B | song | music | 188 | 0.138336 | 0.138336 | +0.000 |
| B → A | pooled | file | 1415 | 0.116515 | 0.109473 | -0.704 |
| B → A | pooled | voice | 1373 | 0.147887 | 0.144919 | -0.297 |
| B → A | pooled | music | 819 | 0.103824 | 0.085477 | -1.835 |
| B → A | korean_speech | file | 596 | 0.052072 | 0.052072 | +0.000 |
| B → A | korean_speech | voice | 596 | 0.052072 | 0.048619 | -0.345 |
| B → A | korean_mix | file | 337 | 0.182436 | 0.160325 | -2.211 |
| B → A | korean_mix | voice | 337 | 0.154211 | 0.145471 | -0.874 |
| B → A | korean_mix | music | 337 | 0.100891 | 0.083087 | -1.780 |
| B → A | external | file | 294 | 0.155952 | 0.155952 | +0.000 |
| B → A | external | voice | 252 | 0.095238 | 0.103175 | +0.794 |
| B → A | external | music | 294 | 0.081633 | 0.061224 | -2.041 |
| B → A | song | file | 188 | 0.117303 | 0.117303 | +0.000 |
| B → A | song | voice | 188 | 0.171302 | 0.171302 | +0.000 |
| B → A | song | music | 188 | 0.153791 | 0.133688 | -2.010 |

## Weight choice and decision

Both-fold fixed-protocol gate: **PASS**. Gate is the unchanged r2 equal-head selection objective: pooled FILE, pooled VOICE, and 0.6×external MUSIC + 0.4×pooled MUSIC. ADS-weighted pooled EER is also recorded in JSON.

| Training fold | Fixed objective sub102r3 → r4 | Held-out selected F/V/M |
|---|---|---|
| A | 0.130454 → 0.128093 | 0.25/0.20/0.35 |
| B | 0.118304 → 0.108440 | 0.30/0.45/0.25 |

Common deployment weights from pooled out-of-fold development predictions: **0.20/0.40/0.30**. These selected-weight metrics reuse held-out labels for tuning and are selection-biased. Fixed-protocol numbers above remain the primary evidence.

Every panel row contributes predictions only from the fold model that did not train its source group during r4. Scoring a two-model average on these same panels would expose each row to one trained model and is not an honest held-out ensemble estimate. If deployed, average the two fold models’ separately calibrated per-file probabilities before the canonical blend; use each fold’s stored pooling parameters. No cross-file normalization is permitted.

team selected the r3-initialized folds as deployment candidates (two-model average or a single fold). No all-data run is queued. A compiled candidate runner is emitted in `r4/r3init/candidate/script.py`; GPU parity, packaging/smoke/runtime validation remain separate team/packager steps. No ZIP has been created or uploaded by this researcher.

## Retained old fold A (diagnostic only)

team changed initialization after this run had exceeded two epochs. It finished six epochs from r2 e09 with freeze-layers 4. It is excluded from the matched-fold deployment decision.

| Panel | Head | sub102r3 | Old A at incumbent weights |
|---|---|---:|---:|
| pooled | file | 0.115668 | 0.109484 |
| pooled | voice | 0.132387 | 0.127026 |
| pooled | music | 0.123571 | 0.128693 |
| korean_speech | file | 0.044988 | 0.048594 |
| korean_speech | voice | 0.044988 | 0.048594 |
| korean_mix | file | 0.129094 | 0.142225 |
| korean_mix | voice | 0.155154 | 0.148548 |
| korean_mix | music | 0.079209 | 0.085810 |
| external | file | 0.190476 | 0.179762 |
| external | voice | 0.095238 | 0.103175 |
| external | music | 0.156463 | 0.183673 |
| song | file | 0.147592 | 0.144701 |
| song | voice | 0.196786 | 0.180715 |
| song | music | 0.138336 | 0.138336 |

## Mixture-validation history (diagnostic only)

| Fold | Epoch | FILE LSE1 | VOICE LSE1 | MUSIC LSE1 | Mean | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 0.098380 | 0.067150 | 0.137124 | 0.100885 | 0.00 |
| A | 1 | 0.098380 | 0.081063 | 0.133126 | 0.104190 | 6.00 |
| A | 2 | 0.098380 | 0.081063 | 0.127499 | 0.102314 | 6.00 |
| A | 3 | 0.088299 | 0.076425 | 0.127499 | 0.097408 | 6.00 |
| B | 0 | 0.098380 | 0.067150 | 0.137124 | 0.100885 | 0.00 |
| B | 1 | 0.101741 | 0.081063 | 0.152377 | 0.111727 | 6.00 |
| B | 2 | 0.103270 | 0.087923 | 0.146750 | 0.112647 | 6.00 |
| B | 3 | 0.109991 | 0.076425 | 0.152377 | 0.112931 | 6.00 |
| legacy_A | 0 | 0.103270 | 0.081063 | 0.137124 | 0.107152 | 0.00 |
| legacy_A | 1 | 0.103270 | 0.087923 | 0.166000 | 0.119064 | 5.18 |
| legacy_A | 2 | 0.109991 | 0.097198 | 0.189249 | 0.132146 | 5.18 |
| legacy_A | 3 | 0.113351 | 0.101836 | 0.198875 | 0.138021 | 5.18 |
| legacy_A | 4 | 0.113351 | 0.101836 | 0.175626 | 0.130271 | 5.18 |
| legacy_A | 5 | 0.106630 | 0.101836 | 0.198875 | 0.135780 | 5.18 |
| legacy_A | 6 | 0.113351 | 0.106473 | 0.208500 | 0.142775 | 5.18 |

## Files and operational checks

- `runs/sub102-nii-ml/r4/r3init/A` and `B` (old `r4/A` retained only as a diagnostic): 3 epoch checkpoints, resume state, exported fp16 overlays, runner-matched held-out panel/mixture scores and evaluation JSON.
- `sub106_r4_results.json`: exact metrics, full 0.05-step curves, pooling, source/code digests and overlay hashes.
- CPU checks passed: fold token exclusions, complete panel keys, masked labels, real-file decoding, identical resumed draws, and all five existing sub102 runner/window/pooling/fallback tests.
- Three additional sub106 CPU tests pass: unchanged sub102r3 runner outside the NII helper block, separately calibrated probability averaging with file-order invariance, and complete fallback/single-fold behavior. Candidate export-to-runner GPU parity is still pending.
- Full audit: all 6341 unique real files match retained expected hashes where supplied; 2,663,577,549 bytes bound in `r4/provenance_receipt.json`. Start/middle/end windows decoded finite for all files, zero exceptions. Eight Jamendo files have silent edge windows; component/presence supervision remains weak at crop level. Partial MP3 seeking emitted mpg123 frame warnings without nonfinite windows. Details: `r4/audio_check.json`.
- Legacy A initial validation reproduced all ten r2 epoch09 metrics exactly. Candidate r3-init parity is checked against the retained r3 epoch12 history. Report prediction/EER calculations match the canonical evaluator for all three heads at weights 0, 0.25 and 1.
- Independent parent-metadata audit found zero hidden source-path/hash/content/speaker overlap against opposing folds and zero parent-source overlap with mixture calibration groups (`scratchpad/sub106_parent_audit.py`). Removed legacy-view matches were music content only (708/816).
- Required r3 initialization, legacy r2 diagnostic and all submitted ZIPs are preserved. Per-track/source licence attributions remain with originals in the original data store.

Risks: modest real-domain sample diversity; weak component occupancy labels on four-second crops; historical panel/initialization exposure; SONICS attribution/ToS residual per register; held-out weight-selection bias; no independent full-data-model or two-model-average EER; no new L4 runtime measurement.

## Deploy / build

Packaged **sub106** from the immutable sub102r3 runner and ZIP. Its NII FT input is fold B alone at its own calibrated probabilities. Final logit weights FILE/VOICE/MUSIC are **0.20/0.40/0.30** from the pooled out-of-fold development curve. The FILE pre-floor blend, original floor, and both presence outputs are unchanged. Fold B beat sub102r3 on its held-out A source groups; deployment weights reuse development panels and remain selection-biased. No competition test data was used for training or tuning.

- ZIP `submissions/sub106.zip`: 8,697,574,886 bytes (<10,000,000,000), SHA256 `4384339cd1f374ca5ccea65c7d60c485d37e19c14af265eb9b4defc77c88d8ae`. Source sub102r3 ZIP SHA256 `8accfe063952f34a0f0aaa782693c7931cf42f3a86a410a2eb4f0bd9fdd32250`. The old sub102r3 overlay was removed and 1 epoch03 export(s) added.
- Runner SHA256 `f17c6ec4de9eed35886a0aa1295175940d92eb6a4c52ea78ff9b7dff63845ffe`; overlay SHA256 B `66cb46db31c7e742e84352095384dec8b1db54957e2d5067b8218be4b65ed462`. Config SHA256 `773fee895b836d9189d3f18602a75a9afec01f81ef2251c1a2858145957d23c5`; source revision `3310b60892e83d210b308b83b1e3d489e14b0145`. `submissions/sub106.build.json` binds every member digest.
- CPU tests 8/8 passed; `validate_submission` accepted 8,697,574,886 compressed bytes and 58 members. Runner/scorer parity across 24 files per included fold: maximum absolute logit differences B 0.
- Offline normal and reversed smoke passed: 22/22 rows, 145.1/127.9 s; exact order-independent probabilities, unchanged presence outputs versus sub102r3, neutral corrupt-file fallback, zero healthy warnings and zero audited network attempts. ZIP and extracted member hashes rechecked after smoke.
- Added-pass benchmark on 150 files: mean excluding first 0.3994 s, p95 0.4241 s; overlay load 1.23 s, peak Torch allocation 4.39 GiB. Projected official runtime 48.01 min; conservative bound 57.61 min (<60). This extrapolates the previous ~40-minute sub100 official runtime plus 1,200 measured additions; it is not an official L4 timing.

Two-fold diagnostic: runner/scorer parity passed, but the 150-file benchmark projected 56.05 min with a conservative 72.05 min bound. This exceeded the 60-minute gate, so team-authorized fold B replaced it. The exact unsubmitted two-fold archive and receipts remain under `r4/r3init/two_fold_diagnostic/` in the original data store.



## r7 folds

Completed 2026-09-28T19:11:33+0900 KST. No ZIP or official upload was made.

Matched r4 A/B manifests, connected-source exclusions, 3 epochs x 1500 steps, seed 406, 35% real draws, optimizer and 517-file non-SONICS calibration were reused exactly. The only model change was the warm start: r5 epoch08 instead of r3 epoch12. Both epoch-zero validation vectors exactly reproduce r5 epoch08. The same fixed F/V/M weights 0.20/0.40/0.30 are used for all three models in this comparison; r4 and sub102r3 were replayed from retained logits at those weights. No per-fold weight search was used for the primary result.

Equal-head mean: pooled FILE, pooled VOICE, MUSIC = 0.6 external + 0.4 pooled; lower is better.

| Train fold / held half | sub102r3 | r3-init r4 | r5-init r7 | r7 - r4 pp | r7 - sub102r3 pp |
|---|---:|---:|---:|---:|---:|
| A / B | 0.140378 | 0.128136 | 0.130407 | +0.227 | -0.997 |
| B / A | 0.122571 | 0.103655 | 0.105276 | +0.162 | -1.730 |
| OOF / both | 0.131667 | 0.114576 | 0.116767 | +0.219 | -1.490 |

### Held-out panel EERs

| Train / held | Panel | Head | n | sub102r3 | r3-init r4 | r5-init r7 | r7-r4 pp |
|---|---|---|---:|---:|---:|---:|---:|
| A / B | pooled | file | 1341 | 0.115668 | 0.106683 | 0.111176 | +0.449 |
| A / B | pooled | voice | 1299 | 0.157055 | 0.132387 | 0.128575 | -0.381 |
| A / B | pooled | music | 785 | 0.126132 | 0.118448 | 0.123571 | +0.512 |
| A / B | korean_speech | file | 556 | 0.052200 | 0.048594 | 0.048594 | +0.000 |
| A / B | korean_speech | voice | 556 | 0.053912 | 0.052200 | 0.053912 | +0.171 |
| A / B | korean_mix | file | 303 | 0.129094 | 0.129094 | 0.131307 | +0.221 |
| A / B | korean_mix | voice | 303 | 0.178179 | 0.155154 | 0.148548 | -0.661 |
| A / B | korean_mix | music | 303 | 0.079209 | 0.079209 | 0.085810 | +0.660 |
| A / B | external | file | 294 | 0.177381 | 0.177381 | 0.192857 | +1.548 |
| A / B | external | voice | 252 | 0.119048 | 0.119048 | 0.134921 | +1.587 |
| A / B | external | music | 294 | 0.163265 | 0.163265 | 0.170068 | +0.680 |
| A / B | song | file | 188 | 0.144701 | 0.136031 | 0.133141 | -0.289 |
| A / B | song | voice | 188 | 0.180715 | 0.164644 | 0.192849 | +2.820 |
| A / B | song | music | 188 | 0.138336 | 0.138336 | 0.138336 | +0.000 |
| B / A | pooled | file | 1415 | 0.116008 | 0.106788 | 0.111652 | +0.486 |
| B / A | pooled | voice | 1373 | 0.163165 | 0.133251 | 0.133251 | +0.000 |
| B / A | pooled | music | 819 | 0.098903 | 0.085477 | 0.085477 | +0.000 |
| B / A | korean_speech | file | 596 | 0.056961 | 0.045166 | 0.045166 | +0.000 |
| B / A | korean_speech | voice | 596 | 0.056961 | 0.055525 | 0.056961 | +0.144 |
| B / A | korean_mix | file | 337 | 0.186343 | 0.160325 | 0.160325 | +0.000 |
| B / A | korean_mix | voice | 337 | 0.184023 | 0.172098 | 0.166136 | -0.596 |
| B / A | korean_mix | music | 337 | 0.100891 | 0.089022 | 0.089022 | +0.000 |
| B / A | external | file | 294 | 0.155952 | 0.142857 | 0.164286 | +2.143 |
| B / A | external | voice | 252 | 0.111111 | 0.111111 | 0.103175 | -0.794 |
| B / A | external | music | 294 | 0.081633 | 0.061224 | 0.061224 | +0.000 |
| B / A | song | file | 188 | 0.117303 | 0.117303 | 0.117303 | +0.000 |
| B / A | song | voice | 188 | 0.159365 | 0.155365 | 0.155365 | +0.000 |
| B / A | song | music | 188 | 0.170017 | 0.137564 | 0.137564 | +0.000 |

Retained two-fold sub106 runner, uncontended 150-file GPU bench: mean 0.8662s/file, p95 0.9189s/file, 40-minute sub100 baseline plus two-fold added pass projects 57.38 min; conservative bound 74.70 min. The earlier contended diagnostic projected 56.05 / 72.05 min. These are local projections, not an official L4 runtime guarantee.

OOF rows are held out only from the new fold continuation. The r5 initialization inherited earlier mixture and panel-adjacent exposure, so these are matched relative comparisons, not pristine unseen-source estimates. The OOF aggregate concatenates each fold's predictions on its opposing half; it is not a two-model average estimate. The all-data sub110 model has no honest held-out half. No competition-test data were used for tuning.

Retained results, score hashes, pooling and checkpoint provenance: `$DATA_DIR\runs\sub102-nii-ml\r7folds\r7fold_results.json`.


## sub111

Completed 2026-09-28T20:14:42.833527 KST. **GO**. No upload performed.


The frozen all-data manifest has 6341 A/B-union real files plus 222 owner-approved SONICS-dependent rows. Two SONICS panel rows with no-derivatives-licensed other parents were excluded. There are no v4 mixtures. Training items SHA256 `33a58baafe1c526d50ba31c8c1db7adf6a07ed0280cefc586ef8c42287ef72c6`. All-data training leaves no honest held-out panel; none was scored or used to select the model. Epoch 4 was fixed in advance.

Warm-start validation exactly reproduced r3 e12. Final training validation mean3_lse1 0.100884753; this did not select the checkpoint. The canonical scorer covered all 1,200 mixture_v2 holdout files. Pooling/affine calibration used the same fixed 517 non-SONICS v2 validation files as sub106; the other 83 val files were scoring-only. Blend weights FILE/VOICE/MUSIC were fixed at 0.20/0.40/0.30 from sub106 OOF development.

Each head on the combined 1,200-file v2 val+test mixture holdout at fixed 0.20/0.40/0.30 must be <= sub106 fold B at the same weights +1 pp; val/test separately are diagnostic.

| split | head | n | sub106 fold B EER | sub111 EER | delta pp |
|---|---|---:|---:|---:|---:|
| val | file | 600 | 0.095020 | 0.088299 | -0.6721 |
| val | voice | 432 | 0.081063 | 0.048599 | -3.2464 |
| val | music | 518 | 0.061750 | 0.056123 | -0.5627 |
| test | file | 600 | 0.111684 | 0.118390 | +0.6706 |
| test | voice | 414 | 0.118335 | 0.082122 | -3.6214 |
| test | music | 535 | 0.046626 | 0.055952 | +0.9325 |
| combined | file | 1200 | 0.103357 | 0.103357 | +0.0000 |
| combined | voice | 846 | 0.096931 | 0.062626 | -3.4305 |
| combined | music | 1053 | 0.054069 | 0.054069 | +0.0000 |

### Package

ZIP `submissions/sub111.zip`, 8,697,574,847 bytes, SHA256 `58a031cbc0d90292839789e6bd86f0e2b2944bead83e2855d4320a94cb52b6c7`. Overlay SHA256 `c6fff8ebdc4b7687b2e2c37115fe858e002fe0e6f4b7d2d282d92f24d0600c6c`; runner SHA256 `99a5b2940c511fc88f13ef16aa4b17858af2634bd4143a476eee9fa35ad98303`; source revision `3310b60892e83d210b308b83b1e3d489e14b0145`.
Five CPU tests, archive validation, 24-file exact runner/scorer parity, normal/reversed offline GPU smokes, unchanged presence and corrupt fallback, exact order independence, zero healthy warnings/network attempts, member and ZIP post-smoke hashes passed. Added pass mean 0.03949s/file, p95 0.05338s; projected runtime 40.81 minutes, conservative 48.98 minutes (<60).

The mixture holdout has been used repeatedly for development; its safety check is not an official-score prediction. This all-data continuation inherits prior r3 mixture exposure, and the panel data were training inputs. No competition-test data or upload were used.


## Longer real-domain fold continuation

Completed 2026-09-29T01:15:33+0900 KST. SUB112_GO. No ZIP or upload was made by this fold experiment.

The retained r3-initialized r4 fold A/B optimizer states at epoch 3 were copied byte-for-byte into isolated r4ext runs. Each continued to epoch 6 with the same fold-local manifests, seed 406, freeze 0, learning rates 5e-6/2e-4, L2-SP 5, 35% real draws, v2 x2 + v3 + safe views, batch 8 and 1500 steps/epoch. Epochs 4, 5 and 6 were scored on the opposing half only. The 517 non-SONICS mixture validation files fit each epoch's canonical pool. All comparisons use fixed FILE/VOICE/MUSIC weights 0.20/0.40/0.30 and the declared equal-head objective (pooled FILE, pooled VOICE, 0.6 external + 0.4 pooled MUSIC).

| Fold epoch | A held B | B held A | Combined OOF objective | vs epoch 3 pp |
|---:|---:|---:|---:|---:|
| 3 | 0.128136 | 0.103655 | 0.114576 | +0.000 |
| 4 | 0.125581 | 0.095843 | 0.110225 | -0.435 |
| 5 | 0.126288 | 0.100686 | 0.111311 | -0.327 |
| 6 | 0.131980 | 0.103441 | 0.116030 | +0.145 |

| Fold / epoch | pooled FILE | pooled VOICE | pooled MUSIC | external MUSIC |
|---|---:|---:|---:|---:|
| A / 3 | 0.106683 | 0.132387 | 0.118448 | 0.163265 |
| A / 4 | 0.106683 | 0.121664 | 0.115887 | 0.170068 |
| A / 5 | 0.104992 | 0.125477 | 0.115887 | 0.170068 |
| A / 6 | 0.109484 | 0.137035 | 0.118448 | 0.170068 |
| B / 3 | 0.106788 | 0.133251 | 0.085477 | 0.061224 |
| B / 4 | 0.099746 | 0.120940 | 0.085477 | 0.054422 |
| B / 5 | 0.104102 | 0.129000 | 0.080556 | 0.061224 |
| B / 6 | 0.106788 | 0.132609 | 0.085477 | 0.061224 |

Best later epoch: 4; OOF gain 0.435 pp. Required gain: 0.2 pp. The original pooled OOF weight optimum is {'file': 0.2, 'voice': 0.4, 'music': 0.3}; its gain over deployed 0.20/0.40/0.30 is 0.000 pp, below the 0.3 pp weight-variant gate.
Decision: **SUB112_GO**. If GO, proportionally continue sub111 by 1 epochs to all-data epoch 5. Full metric matrices, scores and checkpoint hashes: `$DATA_DIR\runs\sub102-nii-ml\r4ext\selection.json`.

The OOF halves are held out only from the new fold continuation; r3 initialization retains earlier mixture exposure. The all-data continuation has no honest panel. Neither the all-data model nor a two-model average is evaluated on these held-out folds.


## sub112

Completed 2026-09-29T01:43:10.198236 KST. **GO**. No upload performed.

Fixed-weight honest-fold continuation selected fold epoch 4 over epoch 3: combined OOF objective 0.114576 to 0.110225, gain 0.435 pp (required >=0.2). Sub111 all-data epoch 4 was then optimizer-resumed for 1 proportional additional epochs to epoch 5, without all-data checkpoint selection.


The frozen all-data manifest has 6341 A/B-union real files plus 222 owner-approved SONICS-dependent rows. Two SONICS panel rows with no-derivatives-licensed other parents were excluded. There are no v4 mixtures. Training items SHA256 `33a58baafe1c526d50ba31c8c1db7adf6a07ed0280cefc586ef8c42287ef72c6`. All-data training leaves no honest held-out panel; none was scored.

Initial validation exactly reproduced r3 e12; copied epoch 4 exactly matches sub111. Final training validation mean3_lse1 0.098857500; this did not select the checkpoint. The canonical scorer covered all 1,200 mixture_v2 holdout files. Pooling/affine calibration used the same fixed 517 non-SONICS v2 validation files as sub106; the other 83 val files were scoring-only. Blend weights FILE/VOICE/MUSIC were fixed at 0.20/0.40/0.30 from sub106 OOF development.

Each head on the combined 1,200-file v2 val+test mixture holdout at fixed 0.20/0.40/0.30 must be <= sub106 fold B at the same weights +1 pp; val/test separately are diagnostic.

| split | head | n | sub106 fold B EER | sub112 EER | delta pp |
|---|---|---:|---:|---:|---:|
| val | file | 600 | 0.095020 | 0.088299 | -0.6721 |
| val | voice | 432 | 0.081063 | 0.053237 | -2.7826 |
| val | music | 518 | 0.061750 | 0.048127 | -1.3624 |
| test | file | 600 | 0.111684 | 0.111684 | +0.0000 |
| test | voice | 414 | 0.118335 | 0.077274 | -4.1061 |
| test | music | 535 | 0.046626 | 0.055952 | +0.9325 |
| combined | file | 1200 | 0.103357 | 0.103357 | +0.0000 |
| combined | voice | 846 | 0.096931 | 0.064996 | -3.1935 |
| combined | music | 1053 | 0.054069 | 0.051301 | -0.2768 |

### Package

ZIP `submissions/sub112.zip`, 8,697,573,972 bytes, SHA256 `1c16b6fb5b78a37c982aee2dc761367e84fcb68f271c35ed7b099c41a689c7e4`. Overlay SHA256 `429258b31b4f4458793a859a0c89fbd7654d98d515348cc096af32e2d6f5558c`; runner SHA256 `347f0270b8598f10b01ccd40de8d308d715a4f05968074567a0d006b6d6bcebd`; source revision `3310b60892e83d210b308b83b1e3d489e14b0145`.
Five CPU tests, archive validation, 24-file exact runner/scorer parity, normal/reversed offline GPU smokes, unchanged presence and corrupt fallback, exact order independence, zero healthy warnings/network attempts, member and ZIP post-smoke hashes passed. Added pass mean 0.04359s/file, p95 0.05468s; projected runtime 40.89 minutes, conservative 49.07 minutes (<60).

The mixture holdout has been used repeatedly for development; its safety check is not an official-score prediction. This all-data continuation inherits prior r3 mixture exposure, and the panel data were training inputs. No competition-test data or upload were used.
