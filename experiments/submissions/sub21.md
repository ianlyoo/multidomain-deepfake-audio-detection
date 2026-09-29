# sub21

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.7655824974; ADS 0.7407222222; CPS 0.9893249735
- Change: FILE head replaced by frozen 17-feature standardized logistic backend fitted on external dev audio only (seed 20260905 C=3); all other heads byte-identical to sub20; per-file fallback to sub20 FILE formula

## Hypothesis

Frozen task-matched FILE fusion transfers its clean .220455 vs .273864 EER edge to the official distribution

## Local evidence recorded before the official result

verify ALL CHECKS PASSED: learned parity 4.996e-16 vs predict_backend on 384 cache rows; holdout FILE EER current .273864 learned .220455; None NaN out-of-range inputs raise to per-file fallback; non-FILE heads identical to base; diff limited to 7 hunks; 37-member clean extraction to D verified all hashes match

## Official interpretation

New best; learned FILE fusion gain +0.0187714286 Score / +0.0208571428 ADS vs sub20; beats fixed-OR control sub22 by +0.0045 Score. Learned-vs-fixed FILE direction decided: learned.

## Decision at the time

Retain learned FILE backend as FILE direction; next challenger from NII/singing/channel evidence on top of sub21 lineage

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
