"""End-to-end calibration pipeline for one mock CSV.

Two selectable engines:
  mode='link'      (default) MML-EM 3PL -> classical features -> learned metric
                   link. Best held-out RMSE.
  mode='xcalibre'  xCalibre-faithful: normal-ogive D=1.702 + floating priors,
                   then a light per-subject metric anchor. Reproduces xCalibre's
                   estimator conventions directly; comparable accuracy.
"""
import os, json
import numpy as np
from loader import load_mock
from calibrate import calibrate_3pl, PriorConfig
from features import classical_stats, build_features
import link as linkmod
from xcalibre import calibrate_xcalibre, XCalConfig
from correction import correct_subject
from scoring import score_subject

DEFAULT_PRIOR = dict(c0=0.25, mu_a=float(np.log(0.80)), sd_a=0.25, sd_g=0.25, sd_d=4.0)
N_POINTS = 61
N_MIN = 50            # below this a module/item is flagged low-confidence
HERE = os.path.dirname(__file__)
LINK_PATH = os.path.join(HERE, 'link_model.json')
XCAL_ANCHOR_PATH = os.path.join(HERE, 'xcalibre_anchor.json')


def prior_from(cfg):
    return PriorConfig(**cfg)


def calibrate_subject(subject_data, prior=None, n_points=N_POINTS):
    prior = prior or prior_from(DEFAULT_PRIOR)
    res = calibrate_3pl(subject_data.responses, prior=prior, n_points=n_points)
    stats = classical_stats(subject_data.responses)
    feat = build_features(res, stats)
    return res, stats, feat


def _apply_xcal_anchor(anchor, subj, res):
    s = anchor['subjects'].get(subj) or next(iter(anchor['subjects'].values()))
    ka, a0 = s['a']; kb, b0 = s['b']
    a = np.clip(res.a * ka + a0, 0.1, 3.0)
    b = np.clip(res.b * kb + b0, -4.0, 4.0)
    return a, b, res.c


def run_mock(path, model=None, prior_cfg=None, n_points=N_POINTS, n_min=N_MIN,
             mode='link', gold=None, mock_id=None, correction='off', strength=1.0,
             score_scale=None):
    """Return dict subject -> list of per-item records with a,b,c and QC.
    mode: 'link' (default) or 'xcalibre'.

    correction: artificial gold-anchored alignment applied per subject when a
    matching, positionally-aligned gold is supplied. One of correction.MODES
    ('off','bias','moment','regress','exact'); strength in [0,1] blends raw->corrected.
    gold: dict (mock_id, subj) -> (J,3) reference array (as gold.load_gold returns).
    mock_id: id used to look up the gold sheet for this mock."""
    prior = prior_from(prior_cfg or DEFAULT_PRIOR)
    if mode == 'link':
        model = model or linkmod.load(LINK_PATH)
    else:
        anchor = json.load(open(XCAL_ANCHOR_PATH))
    data = load_mock(path)
    out = {}
    for subj, sd in data.items():
        stats = classical_stats(sd.responses)
        if mode == 'xcalibre':
            res = calibrate_xcalibre(sd.responses, n_points=n_points)
            a, b, c = _apply_xcal_anchor(anchor, subj, res)
        else:
            res = calibrate_3pl(sd.responses, prior=prior, n_points=n_points)
            feat = build_features(res, stats)
            a, b, c = linkmod.apply_link(model, subj, feat)
        recs = []
        for j, meta in enumerate(sd.items):
            n = int(stats['nadm'][j])
            p = float(stats['p'][j])
            flag = []
            if n < n_min:
                flag.append('low_n')
            if p in (0.0, 1.0) or n < 3:
                flag.append('degenerate')
            recs.append({
                'qid': meta.qid, 'section': meta.section, 'itemtype': meta.itemtype,
                'subject': subj, 'n': n, 'p': round(p, 4),
                'a': round(float(a[j]), 3), 'b': round(float(b[j]), 3),
                'c': round(float(c[j]), 3),
                'a_raw': round(float(res.a[j]), 3), 'b_raw': round(float(res.b[j]), 3),
                'flags': ','.join(flag),
            })
        # artificial gold-anchored correction (referenced mocks only)
        applied = False
        if correction and correction != 'off' and gold is not None:
            gold_arr = gold.get((mock_id, subj)) if mock_id is not None else None
            applied = correct_subject(recs, gold_arr, mode=correction,
                                      strength=strength, n_min=n_min)
        # per-student EAP ability + scaled score, on the engine's native trait metric
        D = 1.702 if mode == 'xcalibre' else 1.0
        sc = score_subject(sd.responses, res.a, res.b, res.c, D=D,
                           n_points=n_points, scale=score_scale)
        scores = {
            'student_id': list(sd.student_ids), 'student_name': list(sd.student_names),
            'theta': [round(float(t), 3) for t in sc['theta']],
            'se': [round(float(s), 3) for s in sc['se']],
            'scaled': [int(v) for v in sc['scaled']],
            'n_answered': [int(v) for v in sc['n_answered']],
            'n_correct': [int(v) for v in sc['n_correct']],
        }
        out[subj] = {
            'items': recs,
            'n_students': sd.n_students,
            'n_iter': res.n_iter,
            'converged': res.converged,
            'correction': f'{correction}@{strength:.2f}' if applied else 'off',
            'scores': scores,
        }
    return out
