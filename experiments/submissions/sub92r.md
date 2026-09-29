# sub92r

- Submitted (KST): 2026-09-27 03:19:30
- Public result: Score 0.8299182116; ADS 0.8122063492; CPS 0.9893249735
- Change: MUSIC = logit_blend(seg_MUSIC17, seg_FT, .5) on PANNs-gated windows; outputs byte-identical to sub92

## Hypothesis

Segment scoring extends past the FT branch in MUSIC (information test)

## Local evidence recorded before the official result

Thin gate PASS: korean_mix MUSIC EER -0.31pp, real-music FP 30->21; external EER +2.72pp (contrary); smoke normal+reversed pass; warm ratio 1.134 vs sub90

## Official interpretation

LOST -.0007714 vs sub90 (ADS -.0008571 => MUSIC EER approx +0.29pp). korean_mix predicted -0.31pp (thin); external predicted +2.72pp: the external sign was right

## Decision at the time

Keep sub90 MUSIC; segment rescoring of MUSIC17 inputs does not transfer. The background-thread overlap held runtime to x1.034 (40m58s)

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
