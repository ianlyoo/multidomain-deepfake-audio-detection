"""sub97b runner -> sub98 runner: the exact sub94 M50 segment pre-floor FILE change on sub97b.

sub98 applies tools/make_sub94_runner.build (the same inserted lines as sub90 -> sub94)
to the pinned sub97b runner. FILE output only; the floor still reads sub70 VOICE and
whole-file MUSIC, so FILE should equal sub94's.

  .venv/Scripts/python.exe tools/make_sub98_runner.py  # writes submission/script_sub98.py
"""
from pathlib import Path
import hashlib

from make_sub94_runner import build as build_sub94

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "83910bc2b01e17373eae6c4aef7c5c8d1c808ab14067177637609fb1bbccb29e"


def build(source: str) -> str:
    return build_sub94(source)


def main():
    src = (ROOT / "submission/script_sub97b.py").read_bytes()
    if hashlib.sha256(src).hexdigest() != SOURCE_SHA:
        raise SystemExit("sub97b source SHA mismatch")
    out = build(src.decode("utf-8")).encode("utf-8")
    compile(out, "script_sub98.py", "exec")
    (ROOT / "submission/script_sub98.py").write_bytes(out)
    print(hashlib.sha256(out).hexdigest())


if __name__ == "__main__":
    main()
