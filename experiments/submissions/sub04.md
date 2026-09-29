# sub04

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.6975967831; ADS 0.6651825397; CPS 0.9893249735
- Change: Kept VOICE_FAKE and MUSIC_FAKE max-pooled while FILE_FAKE used mean-pooled voice/music components

## Hypothesis

Isolate the File EER contribution of mean pooling

## Local evidence recorded before the official result

126 tests passed; GPU smoke; archive CRC structure schema and unique entries passed

## Official interpretation

New best; Score +0.0066857143 and ADS +0.0074285714 versus sub01 imply File EER improved about 0.0148571428; sub02 below sub04 shows combined Voice+Music mean effect is slightly negative

## Decision at the time

Keep sub04 FILE fixed and change only MUSIC_FAKE to mean for exact music attribution

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
