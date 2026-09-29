# sub102 → sub102r3: multi-label mixture fine-tuning

The NII wav2vec-large anti-deepfake backbone was trained with five heads: FILE fake, VOICE fake, MUSIC fake, VOICE present, and MUSIC present. Synthetic render layouts supplied window labels. Fake losses for absent or ambiguous components were masked. Inference used four-second windows with a two-second hop, smooth-max pooling, and separate FILE/VOICE/MUSIC blends.

The r2 checkpoint (sub102) scored **0.8418682116** with F/V/M blend weights 0.20/0.25/0.25. r3 continued from r2, unfroze all transformer layers, and ran twelve more mixture epochs. The selected r3 epoch 12 used weights **0.15/0.10/0.35** and scored **0.8452467831**. It was the final selection. The r2 gain over sub100 was +0.0072714; the r3 gain was another +0.0033786.

The local mixture holdout strongly favored fine-tuning but overstated official transfer by roughly eight times. The panel objective predicted the r2 gain better and under-predicted the r3 gain. `configs/sub102r3_config.json` retains the final pooling coefficients and blend weights. Its local paths are portable placeholders; weights, rendered audio and derived row manifests are excluded. `tools/ft_nii_multilabel.py`, `tools/sub102_prep.py`, `tools/sub102_eval.py`, and `tools/sub102_export.py` preserve the method, but a new training run requires separately acquired inputs.
