"""Compute xCalibre-mode raw calibrations for all mocks once, cache to npz,
then leave-one-out evaluate a light per-subject metric anchor (a & b linear,
c passthrough)."""
import sys, os, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from xcalibre import calibrate_xcalibre
from gold import load_gold

CSV = {os.path.basename(p).split('_-')[1].split('_')[0]: p
       for p in glob.glob('Digital_SAT_Mock_Test_-*.csv')}
GOLD = load_gold()
MOCKS = ['115', '116', '117', '118', '119', '120']
CACHE = os.path.join(os.path.dirname(__file__), '_xcal_raw.npz')


def build_cache():
    store = {}
    t0 = time.time()
    for num in MOCKS:
        for subj, sd in load_mock(CSV[num]).items():
            res = calibrate_xcalibre(sd.responses, n_points=61)
            nadm = (~np.isnan(sd.responses)).sum(0)
            store[f'{num}_{subj}'] = np.column_stack([res.a, res.b, res.c, nadm])
    np.savez(CACHE, **store)
    print(f'cached raw xcalibre calibrations ({time.time()-t0:.0f}s) -> {CACHE}')


def load_cache():
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


def aligned(cache, num, subj):
    k = f'{num}_{subj}'
    return k in cache and (num, subj) in GOLD and cache[k].shape[0] == GOLD[(num, subj)].shape[0]


def fit_anchor(cache, subj, mocks):
    A = np.concatenate([cache[f'{n}_{subj}'][:, 0] for n in mocks])
    B = np.concatenate([cache[f'{n}_{subj}'][:, 1] for n in mocks])
    GA = np.concatenate([GOLD[(n, subj)][:, 0] for n in mocks])
    GB = np.concatenate([GOLD[(n, subj)][:, 1] for n in mocks])
    ka, a0 = np.polyfit(A, GA, 1)
    kb, b0 = np.polyfit(B, GB, 1)
    return float(ka), float(a0), float(kb), float(b0)


def loo(n_min=50):
    cache = load_cache()
    P = {(s, p): [[], []] for s in ('rw', 'math') for p in 'abc'}
    Pt = {'rw': [[], []], 'math': [[], []]}   # b tail
    for subj in ('rw', 'math'):
        al = [n for n in MOCKS if aligned(cache, n, subj)]
        for held in al:
            tr = [n for n in al if n != held]
            ka, a0, kb, b0 = fit_anchor(cache, subj, tr)
            arr = cache[f'{held}_{subj}']; g = GOLD[(held, subj)]
            ea = np.clip(arr[:, 0] * ka + a0, 0.1, 3.0)
            eb = np.clip(arr[:, 1] * kb + b0, -4.0, 4.0)
            ec = arr[:, 2]
            nadm = arr[:, 3]; ok = nadm >= n_min
            p = (g[:, 1] * 0 + 1)  # placeholder
            # p-value approx from n not stored; use difficulty tails via gold-independent proxy skipped
            for pi, par in enumerate('abc'):
                e = [ea, eb, ec][pi]
                P[(subj, par)][0].append(e[ok]); P[(subj, par)][1].append(g[ok, pi])
    print('Leave-one-out (well-adm) — xCalibre mode + light anchor:')
    for subj in ('rw', 'math'):
        row = f'  {subj:4}: '
        for par in 'abc':
            e = np.concatenate(P[(subj, par)][0]); gg = np.concatenate(P[(subj, par)][1])
            row += f'{par} r={np.corrcoef(e, gg)[0, 1]:.2f} rmse={np.sqrt(np.mean((e-gg)**2)):.3f}  '
        print(row)


if __name__ == '__main__':
    if not os.path.exists(CACHE):
        build_cache()
    loo()
