"""Classical item features used by the metric-link model.

Computed identically at link-fit time and at inference time so the frozen
coefficients apply consistently. Features are cheap CTT statistics that carry
information the reference calibration used beyond our MML estimates.
"""
import numpy as np
from scipy.stats import norm


def classical_stats(responses):
    """responses: (N,J) with NaN not-administered.
    Returns dict of per-item arrays: p, rp (point-biserial vs rest score),
    zp (-z of p), zpc (-z of guessing-adjusted p), rbis (biserial), nadm."""
    X = responses
    A = ~np.isnan(X)
    nadm = A.sum(0)
    Xf = np.where(A, X, 0.0)
    ncorrect = Xf.sum(0)
    p = ncorrect / np.maximum(nadm, 1)
    J = X.shape[1]
    rp = np.zeros(J)
    for j in range(J):
        aj = A[:, j]
        if aj.sum() < 3:
            continue
        xj = X[aj, j].astype(float)
        denom = np.maximum(A[aj].sum(1) - 1, 1)
        rest = (np.nansum(X[aj], 1) - xj) / denom
        if xj.std() > 0 and rest.std() > 0:
            rp[j] = np.corrcoef(xj, rest)[0, 1]
    pc = np.clip((p - 0.25) / 0.75, 0.02, 0.98)
    pp = np.clip(p, 0.02, 0.98)
    zpc = -norm.ppf(pc)
    zp = -norm.ppf(pp)
    phi = norm.pdf(zp)
    rbis = np.clip(rp * np.sqrt(pp * (1 - pp)) / np.maximum(np.abs(phi), 1e-6), 0.05, 0.95)
    # continuity-corrected, n-aware guessing-adjusted difficulty: stays informative
    # in the tails (a p=1.0 item is not clipped flat but bounded by its sample size),
    # which the reference b needs to reach extreme (e.g. b=-4) values. zcc3 lets the
    # link bend those tails.
    p_cc = (ncorrect + 0.5) / (nadm + 1.0)
    pc_cc = np.clip((p_cc - 0.25) / 0.75, 1e-3, 1 - 1e-3)
    zcc = -norm.ppf(pc_cc)
    zcc3 = np.sign(zcc) * np.abs(zcc) ** 3
    return dict(p=p, rp=rp, zp=zp, zpc=zpc, rbis=rbis, nadm=nadm, zcc=zcc, zcc3=zcc3)


def build_features(calib, stats):
    """Return dict feature_name -> array, combining MML params and CTT stats."""
    f = dict(a=calib.a, b=calib.b, c=calib.c)
    f.update({k: stats[k] for k in ('p', 'rp', 'zp', 'zpc', 'rbis', 'zcc', 'zcc3')})
    # discrimination×difficulty interaction: a cheap term that sharpens the b link
    # in the tails (a high-biserial very-easy/hard item behaves differently).
    f['inter'] = stats['rbis'] * stats['zcc']
    return f


def design_matrix(feat, cols):
    """Stack named feature columns + intercept."""
    n = len(feat[cols[0]])
    return np.column_stack([feat[c] for c in cols] + [np.ones(n)])
