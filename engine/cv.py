"""Leave-one-mock-out cross-validation of the MML + affine-link calibrator.

For each held-out mock, the affine metric link is fit on the *other* mocks
(pooled, per subject family) and applied to the held-out one. This is the
honest estimate of accuracy on a brand-new mock. n_min flags thin modules.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl, PriorConfig
from tune import CSV, GOLD

MOCKS = ['115', '116', '117', '118', '119', '120']


def calibrate_all(prior, n_points):
    cache = {}
    for num in MOCKS:
        for subj, sd in load_mock(CSV[num]).items():
            res = calibrate_3pl(sd.responses, prior=prior, n_points=n_points)
            nadm = (~np.isnan(sd.responses)).sum(0)
            cache[(num, subj)] = (res.a, res.b, res.c, nadm)
    return cache


def is_aligned(cache, num, subj):
    return (num, subj) in GOLD and (num, subj) in cache and \
        len(cache[(num, subj)][0]) == GOLD[(num, subj)].shape[0]


def fit_link(cache, subj, mocks):
    A = np.concatenate([cache[(n, subj)][0] for n in mocks])
    B = np.concatenate([cache[(n, subj)][1] for n in mocks])
    GA = np.concatenate([GOLD[(n, subj)][:, 0] for n in mocks])
    GB = np.concatenate([GOLD[(n, subj)][:, 1] for n in mocks])
    ka, a0 = np.polyfit(A, GA, 1)
    kb, b0 = np.polyfit(B, GB, 1)
    return ka, a0, kb, b0


def cv(prior, n_points=61, n_min=50, verbose=True):
    cache = calibrate_all(prior, n_points)
    pooled = {'a': [[], []], 'b': [[], []], 'c': [[], []]}  # [est,gold]
    pooled_ok = {'a': [[], []], 'b': [[], []], 'c': [[], []]}  # well-administered
    for subj in ('rw', 'math'):
        aligned = [n for n in MOCKS if is_aligned(cache, n, subj)]
        for held in aligned:
            train = [n for n in aligned if n != held]
            ka, a0, kb, b0 = fit_link(cache, subj, train)
            a, b, c, nadm = cache[(held, subj)]
            g = GOLD[(held, subj)]
            ea, eb = a * ka + a0, b * kb + b0
            ok = nadm >= n_min
            for key, e, gg in [('a', ea, g[:, 0]), ('b', eb, g[:, 1]), ('c', c, g[:, 2])]:
                pooled[key][0].append(e); pooled[key][1].append(gg)
                pooled_ok[key][0].append(e[ok]); pooled_ok[key][1].append(gg[ok])
            if verbose:
                rmse_b = np.sqrt(np.mean((eb - g[:, 1]) ** 2))
                rmse_b_ok = np.sqrt(np.mean((eb[ok] - g[ok, 1]) ** 2)) if ok.any() else float('nan')
                print(f"  {subj:4s} held={held}: b_rmse_all={rmse_b:.3f} b_rmse_welladm={rmse_b_ok:.3f} (thin items={np.sum(~ok)})")

    def summ(d):
        r = {}
        for k in ('a', 'b', 'c'):
            e = np.concatenate(d[k][0]); g = np.concatenate(d[k][1])
            r[k] = (np.corrcoef(e, g)[0, 1], np.sqrt(np.mean((e-g)**2)), np.mean(np.abs(e-g)))
        return r
    return summ(pooled), summ(pooled_ok), cache


if __name__ == '__main__':
    import numpy as np
    priors = {
        'A': PriorConfig(c0=0.25, mu_a=np.log(0.85), sd_a=0.35, sd_g=0.3, sd_d=3.0),
        'B': PriorConfig(c0=0.25, mu_a=np.log(0.80), sd_a=0.25, sd_g=0.25, sd_d=4.0),
        'C': PriorConfig(c0=0.25, mu_a=np.log(0.90), sd_a=0.45, sd_g=0.35, sd_d=2.5),
    }
    for name, pr in priors.items():
        t0 = time.time()
        allr, okr, _ = cv(pr, n_points=61, verbose=False)
        line = f"[{name}] "
        for split, r in [('ALL', allr), ('WELL-ADM', okr)]:
            line += f"{split}: a={r['a'][1]:.3f} b={r['b'][1]:.3f} c={r['c'][1]:.3f} (rb={r['b'][0]:.3f})  "
        print(line, f"{time.time()-t0:.0f}s")
