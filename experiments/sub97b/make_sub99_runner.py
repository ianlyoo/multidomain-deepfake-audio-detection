"""sub97b runner -> sub99 runner: trained window VOICE head (sub99_plan.json arm A_free), VOICE only.

Inside the sub90 segment-VOICE loop, each selected window's fixed composition logit is
replaced by a logistic head over the four window logits (DF, Spectra, NII, Eliya; clip
1e-6). Windows, gate, presence-weighted pooling, the .5 blend with voice38, fallbacks,
FILE (floor reads sub70 VOICE), MUSIC and presences are unchanged. Additive only.

  .venv/Scripts/python.exe tools/make_sub99_runner.py  # writes submission/script_sub99.py
"""
from pathlib import Path
import os
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "83910bc2b01e17373eae6c4aef7c5c8d1c808ab14067177637609fb1bbccb29e"
HEADS = Path(os.environ.get("DATA_DIR", "data")) / "derived/mixture_v2/sub99_heads.json"
HEADS_SHA = None  # bound at generation time and recorded in the runner comment

ANCHOR_CONST = "SUB90_VOICE_BLEND_WEIGHT = 0.5\n"
ANCHOR_LOOP = "        logits.append(np.log(window / (1.0 - window)))\n"
LOOP_ADD = "        logits[-1] = _sub99_window_logit(df, spectra, nii, eliya)\n"


def helpers(coef, intercept, heads_sha):
    return (
        "\n\n# sub99: trained window VOICE head (arm A_free of sub99_plan.json; heads sha256 " + heads_sha + ").\n"
        "SUB99_VOICE_HEAD_COEF = (" + ", ".join(repr(float(c)) for c in coef) + ")\n"
        "SUB99_VOICE_HEAD_INTERCEPT = " + repr(float(intercept)) + "\n\n\n"
        "def _sub99_window_logit(df, spectra, nii, eliya):\n"
        "    \"\"\"Logistic head over the four window logits; replaces the fixed composition.\"\"\"\n"
        "    z = SUB99_VOICE_HEAD_INTERCEPT\n"
        "    for c, value in zip(SUB99_VOICE_HEAD_COEF, (df, spectra, nii, eliya)):\n"
        "        p = float(np.clip(value, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))\n"
        "        z += c * float(np.log(p / (1.0 - p)))\n"
        "    if not np.isfinite(z):\n"
        "        raise ValueError(\"Invalid segment VOICE head logit\")\n"
        "    return z\n")


def build(source: str, heads_raw: bytes) -> str:
    heads = json.loads(heads_raw)["heads"]["A_free"]
    text = source
    for anchor in (ANCHOR_CONST, ANCHOR_LOOP):
        if text.count(anchor) != 1:
            raise SystemExit("anchor count mismatch: " + anchor.strip())
    text = text.replace(ANCHOR_CONST, ANCHOR_CONST + helpers(heads["coef"], heads["intercept"],
                                                             hashlib.sha256(heads_raw).hexdigest()))
    text = text.replace(ANCHOR_LOOP, ANCHOR_LOOP + LOOP_ADD)
    return text


def main():
    src = (ROOT / "submission/script_sub97b.py").read_bytes()
    if hashlib.sha256(src).hexdigest() != SOURCE_SHA:
        raise SystemExit("sub97b source SHA mismatch")
    out = build(src.decode("utf-8"), HEADS.read_bytes()).encode("utf-8")
    compile(out, "script_sub99.py", "exec")
    (ROOT / "submission/script_sub99.py").write_bytes(out)
    print(hashlib.sha256(out).hexdigest())


if __name__ == "__main__":
    main()
