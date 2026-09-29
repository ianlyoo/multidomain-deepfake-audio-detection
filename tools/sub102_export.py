"""sub102: export a training epoch (fp32 trainable tensors) to the deployable fp16 overlay.

Keys stay 'det.<native key>' plus 'heads.weight'/'heads.bias'; the runner overlays them on a copy of
the loaded NII model. All evaluation scores the exported file, so eval and runner see identical weights.

  .venv/Scripts/python.exe tools/sub102_export.py IN.safetensors OUT.safetensors
"""
import hashlib
import json
import sys
from pathlib import Path


def main(src, dst):
    import torch
    from safetensors.torch import load_file, save_file
    state = load_file(src)
    out = {}
    for k, v in state.items():
        if not (k.startswith("det.") or k in ("heads.weight", "heads.bias")):
            raise KeyError(k)
        h = v.to(torch.float16)
        if not torch.isfinite(h).all():
            raise ValueError("non-finite after fp16 cast: " + k)
        out[k] = h.contiguous()
    meta = dict(format="sub102-nii-ml-overlay-v1", source=Path(src).name,
                source_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),
                base_sha256="b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5",
                heads="file_fake,voice_fake,music_fake,voice_present,music_present")
    save_file(out, dst, metadata=meta)
    digest = hashlib.sha256(Path(dst).read_bytes()).hexdigest()
    print(json.dumps(dict(out=dst, tensors=len(out), bytes=Path(dst).stat().st_size, sha256=digest)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
