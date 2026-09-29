# sub08

- Submitted (KST): Not recorded in ledger
- Public result: Score 0.6972539259; ADS 0.6648015873; CPS 0.9893249735
- Change: Replaced only MUSIC_FAKE with ArtifactNet v9.4 mean over first/middle/last four-second raw windows; sub06 File Voice and Presence unchanged

## Hypothesis

A dedicated forensic music detector produces a large official Music EER gain and can later improve File

## Local evidence recorded before the official result

Balanced72 EER 0.1111 AUC 0.9329 versus sub06 music-stem EER 0.2778; full2050 EER 0.1308 AUC 0.9094 with zero errors; real-audio isolation smoke confirms only MUSIC differs

## Official interpretation

Rejected; Score -0.0015428572 and ADS -0.0017142857 versus sub06 imply Music EER worsened exactly 0.005714286; score exactly equals sub02 so local Echoes/FMA evidence did not transfer

## Decision at the time

Do not inject ArtifactNet into FILE; restore sub06 and pivot to a materially different detector family

This report is curated from the public-score ledger. The audio, model package, upload metadata, and hidden test predictions are excluded.
