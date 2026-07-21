"""Fit the metric-link model on all reference mocks and freeze to link_model.json."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl
from features import classical_stats, build_features
from pipeline import prior_from, DEFAULT_PRIOR, N_POINTS, LINK_PATH
from tune import CSV, GOLD
import link as linkmod

MOCKS = ['115', '116', '117', '118', '119', '120']


def main():
    prior = prior_from(DEFAULT_PRIOR)
    samples = []
    for num in MOCKS:
        for subj, sd in load_mock(CSV[num]).items():
            if (num, subj) not in GOLD:
                continue
            g = GOLD[(num, subj)]
            if g.shape[0] != sd.n_items:      # only positionally-aligned datasets
                continue
            res = calibrate_3pl(sd.responses, prior=prior, n_points=N_POINTS)
            feat = build_features(res, classical_stats(sd.responses))
            samples.append({'subject': subj, 'feat': feat, 'gold': g})
    model = linkmod.fit_link_model(samples)
    model['prior'] = DEFAULT_PRIOR
    model['n_points'] = N_POINTS
    model['trained_on'] = MOCKS
    linkmod.save(model, LINK_PATH)
    print('Saved link model ->', LINK_PATH)
    for subj, coef in model['subjects'].items():
        print(f"  {subj}: a_w={np.round(coef['a_w'],3)}  b_w={np.round(coef['b_w'],3)}  c_w={np.round(coef['c_w'],3)}")


if __name__ == '__main__':
    main()
