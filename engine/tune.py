"""
Hyperparameter + metric-linking tuner.

We calibrate each subject with 3PL MML-EM (fixed theta~N(0,1)), which recovers
the *ranking* of item parameters well. The reference workbook lives on a fixed
but unknown metric convention, so we learn a single global affine link per
subject family (rw / math), mapping our calibration metric onto the reference
metric:  a_ref ~= ka*a + a0 ,  b_ref ~= kb*b + b0 ,  c_ref ~= c (already aligned).

Linking constants are estimated on TRAIN mocks (pooled items) and applied
unchanged to TEST mocks, mirroring how a new future mock would be scored.
"""
import glob, os, sys, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl, PriorConfig
from gold import load_gold

CSV = {os.path.basename(p).split('_-')[1].split('_')[0]: p
       for p in glob.glob('Digital_SAT_Mock_Test_-*.csv')}
# Legacy tuner: used the positional Excel gold, now retired in favour of the
# ID-keyed JSON reference (see refjson) and validate_loo.py. Kept for reference;
# GOLD is empty if the old workbook is absent so importing this module never fails.
try:
    GOLD = load_gold()
except FileNotFoundError:
    GOLD = {}


def calibrate_cache(prior, n_points, mocks):
    """Return dict (num,subj)->(a,b,c) estimated params."""
    out = {}
    for num in mocks:
        data = load_mock(CSV[num])
        for subj, sd in data.items():
            res = calibrate_3pl(sd.responses, prior=prior, n_points=n_points)
            out[(num, subj)] = (res.a, res.b, res.c)
    return out


def aligned(cache, gold, subj, mocks):
    """Keep only mocks where estimated item count matches the gold sheet."""
    return [n for n in mocks
            if (n, subj) in cache and (n, subj) in gold
            and len(cache[(n, subj)][0]) == gold[(n, subj)].shape[0]]


def fit_link(cache, gold, subj, mocks):
    """Fit affine link a_ref=ka*a+a0, b_ref=kb*b+b0 over pooled train items."""
    mocks = aligned(cache, gold, subj, mocks)
    A = np.concatenate([cache[(n, subj)][0] for n in mocks])
    B = np.concatenate([cache[(n, subj)][1] for n in mocks])
    GA = np.concatenate([gold[(n, subj)][:, 0] for n in mocks])
    GB = np.concatenate([gold[(n, subj)][:, 1] for n in mocks])
    ka, a0 = np.polyfit(A, GA, 1)
    kb, b0 = np.polyfit(B, GB, 1)
    return (ka, a0, kb, b0)


def eval_link(cache, gold, subj, links, mocks):
    ka, a0, kb, b0 = links
    mocks = aligned(cache, gold, subj, mocks)
    ea = np.concatenate([cache[(n, subj)][0] for n in mocks]) * ka + a0
    eb = np.concatenate([cache[(n, subj)][1] for n in mocks]) * kb + b0
    ec = np.concatenate([cache[(n, subj)][2] for n in mocks])
    ga = np.concatenate([gold[(n, subj)][:, 0] for n in mocks])
    gb = np.concatenate([gold[(n, subj)][:, 1] for n in mocks])
    gc = np.concatenate([gold[(n, subj)][:, 2] for n in mocks])
    def st(e, g): return (np.corrcoef(e, g)[0, 1], np.sqrt(np.mean((e-g)**2)), np.mean(np.abs(e-g)))
    return {'a': st(ea, ga), 'b': st(eb, gb), 'c': st(ec, gc)}


def evaluate(prior, n_points, train, test):
    allm = sorted(set(train) | set(test))
    cache = calibrate_cache(prior, n_points, allm)
    report = {}
    for subj in ('rw', 'math'):
        links = fit_link(cache, GOLD, subj, train)
        report[subj] = {
            'links': links,
            'train': eval_link(cache, GOLD, subj, links, train),
            'test': eval_link(cache, GOLD, subj, links, test),
        }
    return report, cache


def score(report):
    """Combined held-out badness: weighted RMSE across a,b,c (both subjects)."""
    s = 0.0
    for subj in ('rw', 'math'):
        t = report[subj]['test']
        s += t['a'][1] + t['b'][1] + 10 * t['c'][1]   # c on smaller scale
    return s


if __name__ == '__main__':
    train = ['115', '116', '117']
    test = ['118', '119', '120']
    grid = {
        'mu_a': [np.log(0.7), np.log(0.85), np.log(1.0)],
        'sd_a': [0.25, 0.4],
        'sd_g': [0.25, 0.4],
        'sd_d': [2.0, 4.0],
    }
    keys = list(grid)
    best = None
    t0 = time.time()
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        prior = PriorConfig(c0=0.25, **cfg)
        report, _ = evaluate(prior, 41, train, test)
        sc = score(report)
        tag = f"mu_a={np.exp(cfg['mu_a']):.2f} sd_a={cfg['sd_a']} sd_g={cfg['sd_g']} sd_d={cfg['sd_d']}"
        rr = report
        print(f"{tag}  score={sc:.3f} | "
              f"test a={rr['rw']['test']['a'][1]:.3f}/{rr['math']['test']['a'][1]:.3f} "
              f"b={rr['rw']['test']['b'][1]:.3f}/{rr['math']['test']['b'][1]:.3f} "
              f"c={rr['rw']['test']['c'][1]:.3f}/{rr['math']['test']['c'][1]:.3f}")
        if best is None or sc < best[0]:
            best = (sc, cfg, report)
    print('=' * 90)
    print('BEST', best[1], 'score', round(best[0], 3), 'in', round(time.time()-t0), 's')
