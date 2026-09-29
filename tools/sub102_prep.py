"""sub102: per-render component intervals (exact layout) + label-safe speech views -> sub102_items.jsonl.

Each item: path, split, group, n samples, and components [start, end, kind('speech'|'music'), fake(0/1)].
Overlap renders don't record which component comes first; we replay render_local.render's RNG
(Random(seed): speech clip randrange, music clip randrange, then random()<.5) using the
decoded source lengths, and verify clean replays against the rendered FLAC.
"""
from pathlib import Path
import json, random, sys, csv, collections, os
import numpy as np, pandas as pd, soundfile as sf
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deps'))
import render_local_lib as rl  # thin import-safe copy of render_local helpers (load/clip/level)

SR = 16000
CAT = pd.read_csv(ROOT / 'source_catalog.csv', low_memory=False).fillna('').set_index('source_id')
R = [json.loads(l) for l in open(ROOT / 'render_manifest.jsonl', encoding='utf-8')]


def group_of(r):
    ids = [r['speech_id'], r['music_id'], r['partial_fake_id']]
    return '|'.join(CAT.loc[i, 'source_group'] for i in ids if i)


def overlap_order(r):
    rng = random.Random(int(r['seed']))
    n = int(r['samples'])
    speech = rl.load(r['speech_id'])
    rl.clip(speech, n, rng)
    music = rl.load(r['music_id'])
    rl.clip(music, n, rng)
    return 'speech_first' if rng.random() < .5 else 'music_first'


def verify_overlap(r, order):
    """Re-render a clean overlap render and compare to the FLAC."""
    rng = random.Random(int(r['seed']))
    n = int(r['samples'])
    speech = rl.level(rl.clip(rl.load(r['speech_id']), n, rng))
    music = rl.level(rl.clip(rl.load(r['music_id']), n, rng))
    cut = n // 3
    gain = 10 ** (-float(r['snr_db']) / 20)
    x = np.zeros(n, dtype=np.float32)
    first = rng.random() < .5
    assert first == (order == 'speech_first')
    if first:
        x[:2 * cut] = speech[:2 * cut]; x[cut:] += music[cut:] * gain
    else:
        x[:2 * cut] = music[:2 * cut] * gain; x[cut:] += speech[cut:]
    peak = float(np.max(np.abs(x)))
    if peak > .98:
        x = x * (.98 / peak)
    y, _ = sf.read(r['path'], dtype='float32')
    return float(np.max(np.abs(x - y)))


def components(r, order=None):
    n = int(r['samples']); k = r['kind']
    vf, mf = int(r['voice_fake']), int(r['music_fake'])
    S, M = [], []
    if k in ('speech_real', 'speech_fake'):
        S = [(0, n, vf)]
    elif k.startswith('music_'):
        M = [(0, n, mf)]
        if k == 'music_vocal_fake':
            S = [(0, n, 1)]  # fake sung vocals throughout (file label voice_fake=1)
    elif k in ('partial', 'partial_real_music'):
        a = int(round(float(r['splice_start_s']) * SR)); b = a + int(round(float(r['splice_seconds']) * SR))
        S = [(0, a, 0), (a, b, 1), (b, n, 0)]
        if k == 'partial_real_music':
            M = [(0, n, 0)]
    elif k.startswith('sim_'):
        S = [(0, n, vf)]; M = [(0, n, mf)]
    elif k.startswith('seq_'):
        cut = n // 2
        if r['order'] == 'speech_then_music':
            S = [(0, cut, vf)]; M = [(cut, n, mf)]
        else:
            M = [(0, cut, mf)]; S = [(cut, n, vf)]
    elif k.startswith('overlap_'):
        cut = n // 3
        if order == 'speech_first':
            S = [(0, 2 * cut, vf)]; M = [(cut, n, mf)]
        else:
            M = [(0, 2 * cut, mf)]; S = [(cut, n, vf)]
    else:
        raise ValueError(k)
    out = [[a, b, 'speech', f] for a, b, f in S if b > a] + [[a, b, 'music', f] for a, b, f in M if b > a]
    # sanity: file labels equal the OR over components
    assert int(any(c[3] for c in out if c[2] == 'speech')) == vf or k == 'music_vocal_fake', r['id']
    assert int(any(c[3] for c in out if c[2] == 'music')) == mf, r['id']
    return out


def main():
    items = []
    orders = {}
    ov = [r for r in R if r['kind'].startswith('overlap_')]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool:
        for r, o in zip(ov, pool.map(overlap_order, ov)):
            orders[r['id']] = o
    clean = [r for r in ov if r['channel'] == 'clean'][:40]
    errs = [verify_overlap(r, orders[r['id']]) for r in clean]
    print('overlap orders', collections.Counter(orders.values()), 'verify n', len(errs), 'max abs err', max(errs))
    assert max(errs) < 2e-4, errs
    for r in R:
        items.append(dict(id=r['id'], path=r['path'], split=r['split'], source='mixture_v2', kind=r['kind'],
                          channel=r['channel'], group=group_of(r), n=int(r['samples']),
                          comps=components(r, orders.get(r['id'])),
                          file_fake=int(r['file_fake']), voice_fake=int(r['voice_fake']), music_fake=int(r['music_fake']),
                          voice_present=int(r['voice_present']), music_present=int(r['music_present'])))
    # label-safe speech views (sub99/sub60 VOICE rows); FILE/MUSIC labels only where the manifest makes them safe
    index = {}
    for m in (Path(os.environ.get('DATA_DIR', 'data')) / 'derived').glob('*/manifest.csv'):
        with m.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row.get('sha256'):
                    index.setdefault(row['sha256'], row)
    views = [json.loads(l) for l in open(ROOT / 'sub99_train_rows.jsonl', encoding='utf-8')]
    stats = collections.Counter()
    for v in views:
        m = index.get(v['audio_sha256'], {})
        info = sf.info(v['audio_path'])
        n = int(round(info.frames * SR / info.samplerate))
        vl = int(v['voice_label'])
        comps = [[0, n, 'speech', vl]]
        mp = m.get('music_present', '')
        if m.get('domain') == 'speech' or mp == '0':
            file_label = vl; stats['speech_only'] += 1
        elif mp == '1' and m.get('music_fake', '') in ('0', '1'):
            comps.append([0, n, 'music', int(m['music_fake'])])
            file_label = int(m['file_fake']); stats['with_music_' + m['music_fake']] += 1
        else:
            file_label = None; stats['file_masked'] += 1  # unknown background: VOICE only
        items.append(dict(id=v['sample_id'], path=v['audio_path'], split='train', source='sub99_view', kind='view',
                          channel='', group='view:' + v['group'], n=n, comps=comps, file_fake=file_label,
                          voice_fake=vl, music_fake=None, voice_present=1, music_present=None,
                          weight=float(v['weight']), mask_music=('music' not in [c[2] for c in comps] and file_label is None)))
    print('views', stats)
    with open(ROOT / 'sub102_items.jsonl', 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    print('items', len(items), collections.Counter((i['source'], i['split']) for i in items))


if __name__ == '__main__':
    main()
