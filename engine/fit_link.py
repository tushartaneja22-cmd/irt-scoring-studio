"""Fit the metric-link model on all reference mocks and freeze to link_model.json.

Reference truth is the per-mock ID-keyed JSON (`<id>.txt`, see refjson), aligned to
the response matrix by question id (so every item on every mock is used — no
positional/dropped-item misalignment). The math `a` link uses an xCalibre-faithful
slope, so the xcalibre engine is run for the math subject at fit time.
"""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl
from xcalibre import calibrate_xcalibre
from features import classical_stats, build_features
from pipeline import prior_from, DEFAULT_PRIOR, N_POINTS, LINK_PATH
import link as linkmod
import refjson

CSV = {os.path.basename(p).split('_-')[1].split('_')[0]: p
       for p in glob.glob('Digital_SAT_Mock_Test_-*.csv')}


def main():
    prior = prior_from(DEFAULT_PRIOR)
    samples = []
    for num in refjson.MOCKS:
        for subj, sd in load_mock(CSV[num]).items():
            g = refjson.aligned_gold(num, subj, sd.qids)   # (J,3), NaN where unmatched
            keep = ~np.isnan(g[:, 0])
            if keep.sum() == 0:
                continue
            res = calibrate_3pl(sd.responses, prior=prior, n_points=N_POINTS)
            xc = (calibrate_xcalibre(sd.responses, n_points=N_POINTS)
                  if linkmod.needs_xcalibre(None, subj) else None)
            feat = build_features(res, classical_stats(sd.responses), xc=xc)
            # restrict to id-matched items
            feat = {k: np.asarray(v)[keep] for k, v in feat.items()}
            samples.append({'subject': subj, 'feat': feat, 'gold': g[keep]})
            print(f'  {num} {subj}: {int(keep.sum())} items'
                  + ('  (+xcalibre)' if xc is not None else ''))
    model = linkmod.fit_link_model(samples)
    model['prior'] = DEFAULT_PRIOR
    model['n_points'] = N_POINTS
    model['trained_on'] = refjson.MOCKS
    linkmod.save(model, LINK_PATH)
    print('Saved link model ->', LINK_PATH)
    for subj, coef in model['subjects'].items():
        print(f"  {subj}: a_cols={model['a_cols'][subj]}")
        print(f"       a_w={np.round(coef['a_w'], 3)}")


if __name__ == '__main__':
    main()
