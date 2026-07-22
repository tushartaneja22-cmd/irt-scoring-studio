"""Build a cached per-item table (MML + xCalibre estimates, classical features, reference)
aligned by question id, for fast iteration on the `a` link. Writes engine/_acache.npz."""
import sys, os, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl, PriorConfig
from xcalibre import calibrate_xcalibre
from features import classical_stats, build_features
import refjson

PRIOR = PriorConfig(c0=0.25, mu_a=np.log(0.80), sd_a=0.25, sd_g=0.25, sd_d=4.0)


def main():
    rows = []
    for mid in refjson.MOCKS:
        csv = glob.glob(f'*-{mid}_*.csv')[0]
        d = load_mock(csv)
        for subj, sd in d.items():
            t0 = time.time()
            res = calibrate_3pl(sd.responses, prior=PRIOR)
            xres = calibrate_xcalibre(sd.responses)
            st = classical_stats(sd.responses)
            feat = build_features(res, st)
            g = refjson.aligned_gold(mid, subj, sd.qids)
            nadm = (~np.isnan(sd.responses)).sum(0)
            for j in range(sd.n_items):
                rows.append(dict(
                    mid=mid, subj=subj, qid=sd.qids[j], nadm=int(nadm[j]),
                    mml_a=res.a[j], mml_b=res.b[j], mml_c=res.c[j],
                    xc_a=xres.a[j], xc_b=xres.b[j], xc_c=xres.c[j],
                    p=st['p'][j], rp=st['rp'][j], rbis=st['rbis'][j],
                    zcc=st['zcc'][j], zcc3=st['zcc3'][j], zpc=st['zpc'][j], zp=st['zp'][j],
                    g_a=g[j, 0], g_b=g[j, 1], g_c=g[j, 2]))
            print(f'  {mid} {subj}: J={sd.n_items} ({time.time()-t0:.0f}s)')
    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez(os.path.join(os.path.dirname(__file__), '_acache.npz'), **arr)
    print('wrote _acache.npz with', len(rows), 'items')


if __name__ == '__main__':
    main()
