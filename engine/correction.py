"""Artificial (gold-anchored) correction for referenced mocks.

The studio's raw output already ranks items well (b: r~0.96, c near-exact) but on
any single mock it carries a residual deviation from the reference a/b/c that is
two things stacked:

  1. a per-mock offset/scale wobble (esp. `a`, and some mocks' `b`), and
  2. irreducible item-level scatter (b's ~0.4 RMSE is the r~0.96 residual, not bias).

A *learned* global correction cannot fix (1): the offsets flip sign across mocks,
so fitting them on other mocks and applying out-of-sample makes RMSE worse (see
`deviation_analysis`). The only lever that reduces error on a given mock uses that
mock's own reference values — an ARTIFICIAL correction. It therefore requires the
gold workbook to be loaded and applies only to referenced mocks. Use it to
reproduce / audit reference values, not to score a brand-new un-referenced mock.

Modes (increasing aggressiveness), applied per subject × parameter:
  off      identity — raw studio estimate.
  bias     add (mean_gold - mean_est): matches the reference mean, keeps spread & rank.
  moment   z-score to the reference mean AND sd: matches first two moments, keeps rank.
  regress  oracle least-squares  gold ~ k*est + m: minimises RMSE for the given ordering.
  exact    snap to gold: RMSE = 0 (fully artificial; for reproducing the sheet).

`strength` in [0,1] blends raw->corrected, so 0 = off and 1 = full mode.
Moments are computed on well-administered items (n >= n_min) so thin/degenerate
adaptive items don't distort the transform, then the transform applies to all items.
"""
import numpy as np

MODES = ('off', 'bias', 'moment', 'regress', 'exact')
PARAM_RANGES = {'a': (0.1, 3.0), 'b': (-4.0, 4.0), 'c': (0.0, 0.5)}


def _fit_param(est, gold, mode, ok):
    """Return a callable transform f(full_est_array) -> corrected array.
    est/gold are the aligned full arrays; ok is the well-administered mask used
    to estimate the transform constants."""
    e_ok, g_ok = est[ok], gold[ok]
    if mode == 'off' or e_ok.size < 3:
        return lambda x: x.copy()
    if mode == 'exact':
        return lambda x: gold.copy()
    if mode == 'bias':
        shift = float(np.mean(g_ok) - np.mean(e_ok))
        return lambda x: x + shift
    if mode == 'moment':
        se, sg = float(np.std(e_ok)), float(np.std(g_ok))
        me, mg = float(np.mean(e_ok)), float(np.mean(g_ok))
        if se < 1e-9:
            return lambda x: x + (mg - me)
        k = sg / se
        return lambda x: (x - me) * k + mg
    if mode == 'regress':
        A = np.vstack([e_ok, np.ones_like(e_ok)]).T
        k, m = np.linalg.lstsq(A, g_ok, rcond=None)[0]
        return lambda x: k * x + m
    raise ValueError(f'unknown correction mode {mode!r}')


def correct_subject(items, gold_arr, mode='off', strength=1.0, n_min=50):
    """Apply the correction in place to one subject's item records.

    items: list of dicts with 'a','b','c','n' (as built by pipeline.run_mock).
    gold_arr: (J,3) reference a,b,c aligned positionally to items, or None.
    Returns True if applied, False if it could not (no gold / misaligned / off).
    Records keep the raw value in '<p>_uncorrected' and gain '<p>_gold' + 'corrected'.
    """
    if mode == 'off' or gold_arr is None or gold_arr.shape[0] != len(items):
        return False
    strength = float(np.clip(strength, 0.0, 1.0))
    nadm = np.array([it['n'] for it in items])
    ok = nadm >= n_min
    if ok.sum() < 3:                      # too few anchor items -> use all aligned
        ok = np.ones(len(items), bool)
    for pi, p in enumerate('abc'):
        est = np.array([it[p] for it in items], float)
        gold = gold_arr[:, pi].astype(float)
        f = _fit_param(est, gold, mode, ok)
        corr = f(est)
        blended = (1.0 - strength) * est + strength * corr
        lo, hi = PARAM_RANGES[p]
        blended = np.clip(blended, lo, hi)
        for j, it in enumerate(items):
            it[f'{p}_uncorrected'] = it[p]
            it[f'{p}_gold'] = round(float(gold[j]), 3)
            it[p] = round(float(blended[j]), 3)
    for it in items:
        it['corrected'] = f'{mode}@{strength:.2f}'
    return True


def deviation_summary(items, gold_arr, n_min=50):
    """Per-parameter deviation of current item values vs gold (well-admin items).
    Returns dict param -> {bias, rmse, mae, r, sd_est, sd_gold, n}."""
    if gold_arr is None or gold_arr.shape[0] != len(items):
        return None
    nadm = np.array([it['n'] for it in items])
    ok = nadm >= n_min
    if ok.sum() < 3:
        ok = np.ones(len(items), bool)
    out = {}
    for pi, p in enumerate('abc'):
        e = np.array([it.get(f'{p}_uncorrected', it[p]) for it in items], float)[ok]
        g = gold_arr[ok, pi].astype(float)
        r = float(np.corrcoef(e, g)[0, 1]) if e.std() > 0 and g.std() > 0 else float('nan')
        out[p] = dict(bias=float(np.mean(e - g)),
                      rmse=float(np.sqrt(np.mean((e - g) ** 2))),
                      mae=float(np.mean(np.abs(e - g))), r=r,
                      sd_est=float(np.std(e)), sd_gold=float(np.std(g)), n=int(ok.sum()))
    return out
