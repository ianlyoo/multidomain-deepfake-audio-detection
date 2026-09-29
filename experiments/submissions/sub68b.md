# sub68b

- Submitted (KST): 2026-09-24 22:44:06
- Public result: Score 0.8198324974; ADS 0.801; CPS 0.9893249735
- Change: As sub68a but output floor FILE=max(FILE29,vp*VOICE,mp*MUSIC) reads new MUSIC

## Hypothesis

FT MUSIC also helps FILE through mp*MUSIC floor

## Local evidence recorded before the official result

Same model/tests as 68a; GPU22 smoke VOICE/presence identical, FILE changed 8/21, MUSIC 21/21, reverse identical, 0 net

## Official interpretation

WON new best +.0061071 vs 55, +.0026357 vs 68a; floor(new) adds approx +2.9pp ADS via FILE

## Decision at the time

Incumbent=68b; stack further FILE/VOICE changes on 68b

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
