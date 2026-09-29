"""sub102: pooling fit (val) + blend-weight choice (test) for the NII multi-label FT, vs the exact sub100 incumbent.

Incumbent per file is rebuilt from the sub101 replay records (exact sub100 runner code):
  FILE  = floor(max(prefloor FILE', vp*VOICE38, mp*MUSIC68)), FILE' = logit_blend(prefloor FILE, FT_FILE, wF)
  VOICE = published sub90/sub99 VOICE (windows -> sub99 head -> blend .5 with VOICE38) else VOICE38
  MUSIC = published sub80 MUSIC (blend .5 of MUSIC17 with seg FT) else prefloor MUSIC
FT heads blend into the published VOICE/MUSIC (the floor keeps reading VOICE38/MUSIC68) and pre-floor FILE.
CPU only.  .venv/Scripts/python.exe sub102_eval.py --scores $DATA_DIR/r1/scores_epXX --out sub102_eval_epXX.json
"""
from pathlib import Path
import argparse, json, math, sys
import numpy as np
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'src'))
from deepvoicehackathon.metrics import official_eer

EPS = 1e-6
SUB99_COEF = (0.24235476163554562, 0.20492590790397378, 0.1729559943355824, 0.5537457813724879)
SUB99_B = -0.2735022303019513
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 1e3)
GAMMAS = (0.0, 1.0, 2.0)
WS = np.round(np.arange(0, 1.0001, 0.05), 2)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, np.float64)))


def blend(cur, new, w):
    return sig((1 - w) * logit(cur) + w * logit(new))


def eer(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(official_eer(y, s)) if len(np.unique(y)) == 2 else float('nan')


def incumbent(rec):
    """Final sub100 outputs + floor inputs from a sub101 replay record (None if incomplete)."""
    if not rec.get('complete'):
        return None
    row = rec['prefloor_row']
    v38 = float(row['VOICE_FAKE_PROB']); m68 = float(row['MUSIC_FAKE_PROB'])
    vp = float(row['VOICE_PRESENT_PROB']); mp = float(row['MUSIC_PRESENT_PROB'])
    pre = float(row['FILE_FAKE_PROB'])
    voice = v38
    win = rec.get('windows') or []
    if win:
        z = []; w = []
        for df, sp, ni, el, wt in win:
            z.append(SUB99_B + sum(c * float(logit(v)) for c, v in zip(SUB99_COEF, (df, sp, ni, el))))
            w.append(wt)
        voice = round(float(blend(v38, sig(np.average(z, weights=w)), 0.5)), 10)
    music = m68
    if rec.get('sub80_seg_ft') is not None:
        music = round(float(blend(rec['music17'], rec['sub80_seg_ft'], 0.5)), 10)
    file_ = round(max(pre, vp * v38, mp * m68), 10)
    return dict(pre=pre, v38=v38, m68=m68, vp=vp, mp=mp, file=file_, voice=voice, music=music,
                floor_file=float(rec['floor_file']))


def pooled(z, w, tau):
    w = np.asarray(w, np.float64)
    w = w / max(w.sum(), 1e-12)
    zmax = z.max()
    return float(tau * np.log(np.sum(w * np.exp((z - zmax) / tau))) + zmax)


HEADS = dict(file=(0, None, 'file_label', None), voice=(1, 3, 'voice_label', 'voice_present'),
             music=(2, 4, 'music_label', 'music_present'))


def pool_feature(rec, head, tau, gamma):
    h, ph, _, _ = HEADS[head]
    L = np.asarray(rec['logits'], np.float64)
    z = L[:, h]
    w = np.ones(len(z)) if ph is None or gamma == 0 else sig(L[:, ph]) ** gamma + 1e-6
    return pooled(z, w, tau)


def fit_affine(x, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=100.0).fit(np.asarray(x)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def select(recs, head):
    _, _, lab, pres = HEADS[head]
    return [r for r in recs if 'logits' in r and r.get(lab) is not None and (pres is None or r.get(pres) == 1)]


def fit_pooling(recs, head):
    sel = select(recs, head)
    y = np.array([r[HEADS[head][2]] for r in sel])
    best = None
    grid = []
    for tau in TAUS:
        for gamma in (GAMMAS if head != 'file' else (0.0,)):
            x = np.array([pool_feature(r, head, tau, gamma) for r in sel])
            a, b = fit_affine(x, y)
            p = sig(a * x + b)
            bce = float(-np.mean(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1))))
            e = eer(y, sig(x / 8))  # monotone map into [0,1] for the official EER
            grid.append(dict(tau=tau, gamma=gamma, a=a, b=b, eer=e, bce=bce))
            if best is None or (e, bce) < (best['eer'], best['bce']):
                best = grid[-1]
    return best, grid


def ft_prob(rec, head, pool):
    return float(sig(pool['a'] * pool_feature(rec, head, pool['tau'], pool['gamma']) + pool['b']))


def final_outputs(inc, ft, w):
    """Candidate outputs for weights w=(wF, wV, wM); ft=(pF, pV, pM) or None (runner fallback)."""
    if ft is None:
        return inc['file'], inc['voice'], inc['music']
    pre = float(blend(inc['pre'], ft[0], w[0])) if w[0] else inc['pre']
    file_ = max(round(pre, 10), inc['vp'] * inc['v38'], inc['mp'] * inc['m68'])
    voice = float(blend(inc['voice'], ft[1], w[1])) if w[1] else inc['voice']
    music = float(blend(inc['music'], ft[2], w[2])) if w[2] else inc['music']
    return file_, voice, music


def head_eer(rows, head, w, pools):
    idx = dict(file=0, voice=1, music=2)[head]
    _, _, lab, pres = HEADS[head]
    ys, ss = [], []
    for r, inc in rows:
        if r.get(lab) is None or (pres is not None and r.get(pres) != 1):
            continue
        ft = None if 'logits' not in r else tuple(ft_prob(r, h, pools[h]) for h in ('file', 'voice', 'music'))
        ww = [0, 0, 0]; ww[idx] = w
        ys.append(r[lab]); ss.append(final_outputs(inc, ft, ww)[idx])
    return eer(ys, ss), len(ys)


def weight_curve(rows, head, pools):
    return {float(w): head_eer(rows, head, w, pools)[0] for w in WS}


def smooth_argmin(curve):
    ws = sorted(curve); v = np.array([curve[w] for w in ws])
    sm = np.array([np.nanmean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
    return float(ws[int(np.nanargmin(sm))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hold', default='holdout', help="holdout (v2, keys mix2:) or holdout3 (v3, keys mix3:)")
    args = ap.parse_args()
    S = Path(args.scores)
    hold = [json.loads(l) for l in open(S / (args.hold + '.jsonl'), encoding='utf-8')]
    prefix = 'mix3:' if args.hold == 'holdout3' else 'mix2:'
    feats = {}
    for p in ROOT.glob('sub101_features_*.jsonl'):
        for l in open(p, encoding='utf-8'):
            r = json.loads(l); feats[r['key']] = r
    def inc_for(key):
        r = feats.get(key)
        return incumbent(r) if r else None
    val = [r for r in hold if r['split'] == 'val']
    test = [r for r in hold if r['split'] == 'test']
    out = dict(scores=str(S), n_val=len(val), n_test=len(test))
    # standalone FT EERs and pooling (fit on one split, applied on the other)
    for fit_name, fit_set, app_name, app_set in (('val', val, 'test', test), ('test', test, 'val', val)):
        pools = {}
        for head in ('file', 'voice', 'music'):
            best, grid = fit_pooling(fit_set, head)
            pools[head] = best
            sel = select(app_set, head)
            y = [r[HEADS[head][2]] for r in sel]
            out.setdefault('standalone', {})[f'{head}_fit{fit_name}_on{app_name}'] = dict(
                pool=best, eer=eer(y, [ft_prob(r, head, best) for r in sel]),
                eer_mean=eer(y, sig(np.array([pool_feature(r, head, 1e3, 0) for r in sel]) / 8)),
                eer_max=eer(y, sig(np.array([max(np.asarray(r['logits'])[:, HEADS[head][0]]) for r in sel]) / 8)))
        out.setdefault('pools', {})[fit_name] = pools
        rows = [(r, inc_for(prefix + r['key'])) for r in app_set]
        cov = sum(1 for _, i in rows if i is not None)
        rows = [(r, i) for r, i in rows if i is not None]
        out.setdefault('incumbent_coverage', {})[app_name] = cov
        if len(rows) < 100:
            continue
        curves = {h: weight_curve(rows, h, pools) for h in ('file', 'voice', 'music')}
        out.setdefault('curves', {})[f'pool{fit_name}_weights{app_name}'] = curves
        out.setdefault('chosen', {})[f'pool{fit_name}_weights{app_name}'] = {h: smooth_argmin(c) for h, c in curves.items()}
    # panels (information only): pooling from val, weights from val->test choice
    pp = S / 'panels.jsonl'
    if pp.exists() and 'val' in out.get('pools', {}):
        pools = out['pools']['val']
        panels = [json.loads(l) for l in open(pp, encoding='utf-8')]
        chosen = out.get('chosen', {}).get('poolval_weightstest', dict(file=.3, voice=.3, music=.3))
        res = {}
        for name in ('korean_speech', 'korean_mix', 'external', 'song'):
            rows = [(r, inc_for(r['key'])) for r in panels if r['panel'] == name]
            rows = [(r, i) for r, i in rows if i is not None]
            if not rows:
                continue
            res[name] = dict(n=len(rows))
            for h in ('file', 'voice', 'music'):
                base, n = head_eer(rows, h, 0.0, pools)
                if n == 0:
                    continue
                res[name][h] = dict(n=n, base=base, chosen_w=chosen[h], at_chosen=head_eer(rows, h, chosen[h], pools)[0],
                                    curve={float(w): head_eer(rows, h, w, pools)[0] for w in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)})
                sel = select([r for r, _ in rows], h)
                res[name][h]['ft_alone'] = eer([r[HEADS[h][2]] for r in sel], [ft_prob(r, h, pools[h]) for r in sel]) if sel else None
        out['panels'] = res
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('incumbent_coverage', 'chosen') if k in out}, indent=1))
    for k, v in out.get('standalone', {}).items():
        print(k, round(v['eer'], 4), 'mean', round(v['eer_mean'], 4), 'max', round(v['eer_max'], 4), v['pool']['tau'], v['pool']['gamma'])
    for name, r in out.get('panels', {}).items():
        print(name, {h: (round(x['base'], 4), round(x['at_chosen'], 4)) for h, x in r.items() if isinstance(x, dict)})


if __name__ == '__main__':
    main()
