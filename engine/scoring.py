"""Per-student ability scoring from calibrated 3PL item parameters.

Given an item pool's parameters and a student's responses, estimate the student's
latent ability theta by EAP (expected a-posteriori) on the same fixed N(0,1) trait
used for calibration, then map theta onto a Digital-SAT-style section scale.

EAP is the posterior mean of theta:
    theta_hat_i = sum_q  theta_q * post_iq ,   post_iq ∝ w_q * prod_j P_j(theta_q)^x_ij (1-P_j)^(1-x_ij)
over administered items j only. It is robust for adaptive data (missing-by-design
items simply drop out of the product) and, unlike MLE, always returns a finite score
even for all-correct / all-incorrect students.

Scaled score: theta is standardised on N(0,1), so the section score is a linear
norm-referenced map  scaled = round(mean + sd*theta)  clamped to [lo,hi] and rounded
to the nearest `step`. Defaults (mean=500, sd=100, [200,800], step=10) mirror the SAT
section scale. This is a transparent norm-referenced conversion, NOT the College
Board's official (proprietary, form-specific) raw-to-scaled table.
"""
import numpy as np
from scipy.special import expit

SECTION = dict(mean=500.0, sd=100.0, lo=200.0, hi=800.0, step=10.0)


def _grid(n_points=61, lo=-6.0, hi=6.0):
    theta = np.linspace(lo, hi, n_points)
    w = np.exp(-0.5 * theta ** 2)
    w /= w.sum()
    return theta, w


def eap_theta(responses, a, b, c, D=1.0, n_points=61):
    """responses (N,J) with NaN not-administered; a,b,c item params (length J).
    Returns dict with per-student arrays: theta, se, n_answered, n_correct.
    P_j(theta) = c + (1-c) * sigmoid(D * a * (theta - b))."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    A = (~np.isnan(responses)).astype(float)          # (N,J) administered
    Xf = np.where(A > 0, responses, 0.0)              # (N,J) correct (0 where missing)
    theta, w = _grid(n_points)
    z = D * a[:, None] * (theta[None, :] - b[:, None])  # (J,Q)
    P = np.clip(c[:, None] + (1 - c[:, None]) * expit(z), 1e-9, 1 - 1e-9)
    logP = np.log(P); log1mP = np.log1p(-P)
    # person log-likelihood at each node: (N,Q)
    LL = Xf @ logP + (A - Xf) @ log1mP
    LL -= LL.max(axis=1, keepdims=True)
    post = np.exp(LL) * w[None, :]
    post /= post.sum(axis=1, keepdims=True)
    theta_hat = post @ theta
    var = post @ (theta ** 2) - theta_hat ** 2
    return dict(theta=theta_hat, se=np.sqrt(np.maximum(var, 0.0)),
                n_answered=A.sum(1).astype(int), n_correct=Xf.sum(1).astype(int))


def to_scaled(theta, mean=None, sd=None, lo=None, hi=None, step=None):
    """Map standardised theta -> section score, clamped & rounded to `step`."""
    s = SECTION
    mean = s['mean'] if mean is None else mean
    sd = s['sd'] if sd is None else sd
    lo = s['lo'] if lo is None else lo
    hi = s['hi'] if hi is None else hi
    step = s['step'] if step is None else step
    raw = mean + sd * np.asarray(theta, float)
    snapped = np.round(raw / step) * step
    return np.clip(snapped, lo, hi)


def score_subject(responses, a, b, c, D=1.0, n_points=61, scale=None):
    """Convenience: EAP + scaled score for one subject. scale overrides SECTION."""
    scale = scale or {}
    r = eap_theta(responses, a, b, c, D=D, n_points=n_points)
    r['scaled'] = to_scaled(r['theta'], **scale)
    return r
