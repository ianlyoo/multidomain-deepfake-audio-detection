# sub07

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.6897967831; ADS 0.6565158730; CPS 0.9893249735
- Change: Replaced only sub06 FILE_FAKE with mean-pooled raw-mixture DF-Arena while preserving sub06 Voice Music and Presence outputs

## Hypothesis

Test the raw-input by mean-pooling interaction and whether avoiding Demucs improves the 45-percent File target

## Local evidence recorded before the official result

126 tests passed; GPU smoke; archive CRC structure schema and unique entries passed; isolated raw-path fallback

## Official interpretation

Strong rejection; Score fell 0.009 and ADS fell 0.010 versus sub06 so File EER worsened exactly 2 percentage points; raw mean also lost to raw max after accounting for Voice mean

## Decision at the time

Restore sub06 and stop spending milestones on the existing raw DF-Arena pooling grid; require a new detector training signal or materially different ensemble for sub08

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
