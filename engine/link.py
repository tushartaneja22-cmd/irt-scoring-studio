"""Fit and apply the metric-link model that maps calibration output onto the
reference (ID-keyed) a/b/c metric.

The link is a small linear model per subject family (rw / math):
    a_ref = w_a . [features_a, 1]
    b_ref = w_b . [features_b, 1]
    c_ref = w_c . [features_c, 1]
Coefficients are estimated once on all reference mocks and frozen to JSON, then
applied to any new mock.

Feature columns for `a` are chosen per subject, because the two subjects behave
differently:
  * rw    — the reference couples discrimination to difficulty, so `a` is best
            recovered from the MML slope plus classical discrimination proxies and
            guessing-adjusted difficulty (`b`, `zpc`). All cheap, MML-only.
  * math  — the reference discrimination ranking is recovered markedly better by
            the xCalibre-faithful estimator (normal-ogive + floating priors), so
            math `a` adds the `xc_a` feature. That requires running the xcalibre
            engine for the math subject at inference (see pipeline.needs_xcalibre).
`b` and `c` share one feature set across subjects.
"""
import json
import numpy as np
from features import design_matrix

# per-subject `a` feature columns (see module docstring). 'a'/'b'/'c' are the MML
# estimates; 'xc_a' is the xCalibre-faithful slope; the rest are classical stats.
A_COLS = {
    'rw':   ['a', 'rbis', 'rp', 'b', 'zpc'],
    'math': ['a', 'xc_a'],
}
A_COLS_DEFAULT = ['a', 'rbis', 'rp']          # fallback for an unknown subject
B_COLS = ['b', 'zcc', 'zcc3', 'zpc', 'rbis', 'rp', 'inter']
C_COLS = ['c']
B_CLIP = (-4.0, 4.0)   # reference difficulties are bounded at the tails


def a_cols_for(subject, model=None):
    """Feature columns used for `a` on this subject (model may override defaults)."""
    if model is not None:
        ac = model.get('a_cols', {})
        if subject in ac:
            return ac[subject]
    return A_COLS.get(subject, A_COLS_DEFAULT)


def needs_xcalibre(model, subject):
    """True if this subject's `a` link references an xCalibre feature (xc_*),
    so the pipeline must run the xcalibre engine for it."""
    return any(str(c).startswith('xc_') for c in a_cols_for(subject, model))


def fit_linear(feat_list, gold_list, cols):
    """Least squares over pooled items from multiple datasets."""
    X = np.vstack([design_matrix(f, cols) for f in feat_list])
    y = np.concatenate(gold_list)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w.tolist()


def fit_link_model(samples):
    """samples: list of dicts {subject, feat, gold(a,b,c array)}.
    Returns model dict with per-subject coefficients and the a-columns used."""
    model = {'a_cols': {}, 'b_cols': B_COLS, 'c_cols': C_COLS,
             'b_clip': list(B_CLIP), 'subjects': {}}
    for subj in ('rw', 'math'):
        sub = [s for s in samples if s['subject'] == subj]
        if not sub:
            continue
        feats = [s['feat'] for s in sub]
        acols = a_cols_for(subj)
        model['a_cols'][subj] = acols
        model['subjects'][subj] = {
            'a_w': fit_linear(feats, [s['gold'][:, 0] for s in sub], acols),
            'b_w': fit_linear(feats, [s['gold'][:, 1] for s in sub], B_COLS),
            'c_w': fit_linear(feats, [s['gold'][:, 2] for s in sub], C_COLS),
        }
    return model


def apply_link(model, subject, feat):
    """Return linked (a, b, c) arrays for one dataset. Falls back to the other
    subject's coefficients if the requested subject was not fitted."""
    subj = subject if subject in model['subjects'] else next(iter(model['subjects']))
    coef = model['subjects'][subj]
    a = design_matrix(feat, a_cols_for(subj, model)) @ np.array(coef['a_w'])
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
