# sub23

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.7627824974; ADS 0.7376111111; CPS 0.9893249735
- Change: Submitted VOICE NII20 to40 logit from saved preNII; winning FILE backend and all its inputs stay at sub21 NII20; MUSIC and presence unchanged

## Hypothesis

Higher NII contribution retains additional official VOICE headroom while protecting the learned FILE gain

## Local evidence recorded before the official result

Exact500 EER ties .0120094154; AUC .9944276313 to .9974219788; paired49speaker uncertainty crosses0; 514tests pass1skip;37CRC/SHA clean extraction; actual22fixtureGPU 88nonVOICEvalues exactlysame;VOICE21rowschanged;true reverse110valuesexact; no healthy-input warnings

## Official interpretation

Lost to sub21 by .0028000000 Score and .0031111111 ADS; isolated VOICE EER worsened about 1.55556 percentage points; CPS unchanged; runtime 1662s is 58s longer

## Decision at the time

Retain sub21 NII20 and learned FILE; reject tested VOICE40 variant rather than whole family; prioritize new-source channel FILE evidence

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
