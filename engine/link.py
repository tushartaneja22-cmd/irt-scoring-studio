"""Fit and apply the metric-link model that maps MML calibration output onto
the reference (gold-standard) a/b/c metric.

The link is a small linear model per subject family (rw / math):
    a_ref = w_a . [features_a, 1]
    b_ref = w_b . [features_b, 1]
    c_ref = w_c . [features_c, 1]
Coefficients are estimated once on all available reference mocks and frozen to
JSON, then applied to any new mock.
"""
import json
import numpy as np
from features import design_matrix

# feature columns per parameter (shared across subjects)
A_COLS = ['a']
B_COLS = ['b', 'zcc', 'zcc3', 'zpc', 'rbis']
C_COLS = ['c']
B_CLIP = (-4.0, 4.0)   # reference difficulties are bounded at the tails


def fit_linear(feat_list, gold_list, cols):
    """Least squares over pooled items from multiple datasets."""
    X = np.vstack([design_matrix(f, cols) for f in feat_list])
    y = np.concatenate(gold_list)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w.tolist()


def fit_link_model(samples):
    """samples: list of dicts {subject, feat, gold(a,b,c array)}.
    Returns model dict with per-subject coefficients."""
    model = {'a_cols': A_COLS, 'b_cols': B_COLS, 'c_cols': C_COLS,
             'b_clip': list(B_CLIP), 'subjects': {}}
    for subj in ('rw', 'math'):
        sub = [s for s in samples if s['subject'] == subj]
        if not sub:
            continue
        feats = [s['feat'] for s in sub]
        model['subjects'][subj] = {
            'a_w': fit_linear(feats, [s['gold'][:, 0] for s in sub], A_COLS),
            'b_w': fit_linear(feats, [s['gold'][:, 1] for s in sub], B_COLS),
            'c_w': fit_linear(feats, [s['gold'][:, 2] for s in sub], C_COLS),
        }
    return model


def apply_link(model, subject, feat):
    """Return linked (a, b, c) arrays for one dataset. Falls back to the other
    subject's coefficients if the requested subject was not fitted."""
    subj = subject if subject in model['subjects'] else next(iter(model['subjects']))
    coef = model['subjects'][subj]
    a = design_matrix(feat, model['a_cols']) @ np.array(coef['a_w'])
    b = design_matrix(feat, model['b_cols']) @ np.array(coef['b_w'])
    c = design_matrix(feat, model['c_cols']) @ np.array(coef['c_w'])
    # keep parameters in sane ranges
    a = np.clip(a, 0.1, 3.0)
    c = np.clip(c, 0.0, 0.5)
    b = np.clip(b, *model.get('b_clip', B_CLIP))
    return a, b, c


def save(model, path):
    with open(path, 'w') as fh:
        json.dump(model, fh, indent=2)


def load(path):
    with open(path) as fh:
        return json.load(fh)
