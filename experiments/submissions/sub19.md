# sub19

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.7444967831; ADS 0.7172936508; CPS 0.9893249735
- Change: Kept every sub18 output except VOICE_FAKE_PROB receives a fixed 80 percent sub18 plus 20 percent NII logit blend

## Hypothesis

The orthogonal NII wav2vec anti-deepfake ranking can correct unseen-generator VOICE errors while a bounded blend preserves the officially strong DF-Arena and Spectra base

## Local evidence recorded before the official result

Exact package-driven In-the-Wild 500-file EER improved 0.0240 to 0.0120094 and AUC 0.9892 to 0.9944276; 5000 grouped bootstrap median EER improved 0.0218 to 0.0123; exact108 was non-inferior; raw1000 improved 0.010 to 0.008; 349 tests passed and 1 skipped; 14 staged and 15 final manifest hashes passed; final ZIP CRC roots size schema finiteness bounds clean extraction GPU smoke and byte-identical output passed

## Official interpretation

New best; Score gained 0.0052000000 and ADS 0.0057777778 versus sub18; isolated VOICE EER improved approximately 2.8888889 percentage points; runtime increased 45 seconds

## Decision at the time

Use sub19 as incumbent; audit FILE transfer and task-matched validation before choosing next milestone

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
