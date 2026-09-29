"""sub90 runner -> sub97b runner: the segment FT-MUSIC branch reads its own pinned head.

sub97b (exploratory, post-hoc): the sub80 PANNs-gated 4 s window branch
(_sub80_final_music) scores windows with model/ft_music_seg_sub97b.json, a
window-bag retrain of release-v2 with a training-data-recentred bias
($DATA_DIR/derived/mixture_v2/sub97b_plan.json). The whole-file
FT path (release-v2, FT_MUSIC_PATH) that feeds FILE and the floor is unchanged,
so FILE, VOICE and presences are byte-identical to sub90. A segment-head load
failure falls back to the release-v2 head (exact sub90 MUSIC). The only
non-additive edit is the model argument of the _sub80_final_music call.

  .venv/Scripts/python.exe tools/make_sub97b_runner.py  # writes submission/script_sub97b.py
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "3f0279e9ba79f78a89573bb0271274524f0df6fa4c7f4c0ed9a7091dd9e81331"
SEG_MODEL_SHA = "7fcc13317c6b776d74b5af31278254027654510338dd4c5a4999f205033a9e10"
SEG_MEMBER = "model/ft_music_seg_sub97b.json"

CONSTANT_ANCHOR = 'FT_MUSIC_FLOOR = "new"\n'
CONSTANTS = (
    '# sub97b: segment-only FT MUSIC head (window-bag retrain, recentred bias).\n'
    'FT_MUSIC_SEG_PATH = MODEL_DIR / "ft_music_seg_sub97b.json"\n'
    f'FT_MUSIC_SEG_SHA256 = "{SEG_MODEL_SHA}"\n')
LOADER_ANCHOR = 'def _sub68_logit_blend(current, score, weight):\n'
LOADER = '''def load_ft_music_seg_model():
    """sub97b: digest-pinned, size-bounded segment-branch FT MUSIC head."""
    import hashlib
    with FT_MUSIC_SEG_PATH.open("rb") as handle:
        raw = handle.read(FT_MUSIC_MAX_BYTES + 1)
    if len(raw) > FT_MUSIC_MAX_BYTES:
        raise ValueError("segment FT MUSIC model exceeds size bound")
    if hashlib.sha256(raw).hexdigest() != FT_MUSIC_SEG_SHA256:
        raise ValueError("segment FT MUSIC model digest mismatch")
    return ftm_prepare(json.loads(raw))


'''
LOAD_ANCHOR = '        ft_music_model = None\n\n    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):\n'
LOAD = (
    '        ft_music_model = None\n\n'
    '    # sub97b: the segment branch reads its own head; failure keeps release-v2 (exact sub90).\n'
    '    try:\n'
    '        ft_music_seg_model = load_ft_music_seg_model()\n'
    '    except (Exception, SystemExit) as ft_music_seg_error:\n'
    '        print(f"WARNING: segment MUSIC head failed to load: {ft_music_seg_error}; "\n'
    '              "retaining release-v2 segment head", file=sys.stderr)\n'
    '        ft_music_seg_model = ft_music_model\n\n'
    '    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):\n')
CALL_OLD = '                        ft_music_model, audio_path, music17, segment_music)\n'
CALL_NEW = '                        ft_music_seg_model, audio_path, music17, segment_music)\n'


def build(source: str) -> str:
    text = source
    for anchor in (CONSTANT_ANCHOR, LOADER_ANCHOR, LOAD_ANCHOR, CALL_OLD):
        if text.count(anchor) != 1:
            raise SystemExit("anchor count mismatch: " + anchor.strip()[:80])
    text = text.replace(CONSTANT_ANCHOR, CONSTANT_ANCHOR + CONSTANTS)
    text = text.replace(LOADER_ANCHOR, LOADER + LOADER_ANCHOR)
    text = text.replace(LOAD_ANCHOR, LOAD)
    text = text.replace(CALL_OLD, CALL_NEW)
    return text


def main():
    src = (ROOT / "submission/script_sub90.py").read_bytes()
    if hashlib.sha256(src).hexdigest() != SOURCE_SHA:
        raise SystemExit("sub90 source SHA mismatch")
    out = build(src.decode("utf-8")).encode("utf-8")
    compile(out, "script_sub97b.py", "exec")
    (ROOT / "submission/script_sub97b.py").write_bytes(out)
    print(hashlib.sha256(out).hexdigest())


if __name__ == "__main__":
    main()
