# sub02

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.6972539259; ADS 0.6648015873; CPS 0.9893249735
- Change: Changed DF-Arena segment aggregation from max to mean for voice and music; file remained presence-gated component maximum

## Hypothesis

Removing segment-count and duration bias improves ranking

## Local evidence recorded before the official result

126 tests passed; archive CRC structure and unique entries passed

## Official interpretation

New best; Score +0.0063428571 and ADS +0.0070476190 versus sub01 with unchanged CPS; runtime 21m19s

## Decision at the time

Decompose mean-pooling gain by voice versus music and combine with raw FILE_FAKE signal

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
