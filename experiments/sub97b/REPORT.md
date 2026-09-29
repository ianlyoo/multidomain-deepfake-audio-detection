# sub97b: window-matched music head

The hypothesis was that a music classifier trained on the same four-second windows seen at inference would transfer better than a whole-file head. A retrained segment head was blended into the music output while FILE, VOICE, and presence outputs stayed fixed. An exploratory calibration corrected an early real-music false-positive regression.

Official public score: **0.8368610688** (ADS 0.8199206349, CPS 0.9893249735), about +0.00617 over sub90. External music EER improved approximately 2.38 percentage points in the local panel. The final runner retained this branch, with later FILE/VOICE and mixture fine-tune additions. The generator script is preserved here as lineage code; the source runner and weights are external to this public repository.
