"""Honest leave-one-mock-out validation through the production link code path.

Rebuilds per-(mock,subject) feature dicts from the cached estimate table
(_acache.npz) and runs link.fit_link_model / apply_link exactly as the app would,
holding each mock out. Reports a/b/c accuracy vs the ID-keyed JSON reference."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import link as linkmod
import refjson

_CACHE = os.path.join(os.path.dirname(__file__), '_acache.npz')
if not os.path.exists(_CACHE):
    import _build_cache
    _build_cache.main()
C = dict(np.load(_CACHE, allow_pickle=True))
MOCKS = refjson.MOCKS

# map cache column -> feature name expected by link/features
FEATMAP = {'a': 'mml_a', 'b': 'mml_b', 'c': 'mml_c', 'xc_a': 'xc_a', 'xc_b': 'xc_b',
           'xc_c': 'xc_c', 'p': 'p', 'rp': 'rp', 'rbis': 'rbis', 'zcc': 'zcc',
           'zcc3': 'zcc3', 'zpc': 'zpc', 'zp': 'zp'}


def feat_for(mid, subj):
    m = (C['mid'] == mid) & (C['subj'] == subj)
    f = {fn: C[col][m].astype(float) for fn, col in FEATMAP.items()}
    f['inter'] = f['rbis'] * f['zcc']
    gold = np.column_stack([C['g_a'][m], C['g_b'][m], C['g_c'][m]]).astype(float)
    nadm = C['nadm'][m].astype(int)
    return f, gold, nadm


def run():
    cells = {(subj, k): [[], [], []] for subj in ('rw', 'math') for k in ('a', 'b', 'c')}  # est,gold,nadm
    for held in MOCKS:
        train = [m for m in MOCKS if m != held]
        samples = []
        for subj in ('rw', 'math'):
            for m in train:
                f, g, _ = feat_for(m, subj)
                samples.append({'subject': subj, 'feat': f, 'gold': g})
        model = linkmod.fit_link_model(samples)
        for subj in ('rw', 'math'):
            f, g, nadm = feat_for(held, subj)
            a, b, c = linkmod.apply_link(model, subj, f)
            for k, e, gg in [('a', a, g[:, 0]), ('b', b, g[:, 1]), ('c', c, g[:, 2])]:
                cells[(subj, k)][0].append(e)
                cells[(subj, k)][1].append(gg)
                cells[(subj, k)][2].append(nadm)

    def summ(subj, k, well=False):
        e = np.concatenate(cells[(subj, k)][0]); g = np.concatenate(cells[(subj, k)][1])
        n = np.concatenate(cells[(subj, k)][2])
        ok = n >= 50 if well else np.ones_like(n, bool)
        e, g = e[ok], g[ok]
        return np.corrcoef(e, g)[0, 1], np.sqrt(np.mean((e - g) ** 2)), len(e)

    print(f'{"":6}{"a r":>8}{"a rmse":>9}{"b r":>8}{"b rmse":>9}{"c r":>8}{"c rmse":>9}{"n":>6}')
    for well, tag in [(False, 'ALL'), (True, '>=50')]:
        print(f'-- {tag} --')
        for subj in ('rw', 'math'):
            ar, ae, n = summ(subj, 'a', well)
            br, be, _ = summ(subj, 'b', well)
            cr, ce, _ = summ(subj, 'c', well)
            print(f'{subj:6}{ar:8.3f}{ae:9.3f}{br:8.3f}{be:9.3f}{cr:8.3f}{ce:9.3f}{n:6d}')


if __name__ == '__main__':
    run()
