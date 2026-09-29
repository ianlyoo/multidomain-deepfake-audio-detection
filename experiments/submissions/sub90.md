# sub90

- Submitted (KST): 2026-09-27 00:59:45
- Public result: Score 0.8306896402; ADS 0.8130634921; CPS 0.9893249735
- Change: VOICE output = logit-blend .5 of old VOICE and speech-weighted pooled 4-model VOICE on PANNs speech-present vocal-stem windows

## Hypothesis

Segment scoring transfers to VOICE as it did for MUSIC

## Local evidence recorded before the official result

Panel gate PASS: korean_speech VOICE EER 5.82->5.04, real-speech FP 30->22; korean_mix -2.66pp; smoke normal+reversed pass; runtime ratio 1.16 mean vs sub80

## Official interpretation

WON NEW BEST +.0016000 vs sub80 (ADS +.0017778 => VOICE EER approx -0.89pp); korean_speech panel predicted -0.78pp

## Decision at the time

Incumbent=sub90; runtime 39m37s (x1.33 vs sub80) so runtime headroom is now the constraint; sub92 (MUSIC17 seg) held on the runtime condition

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
