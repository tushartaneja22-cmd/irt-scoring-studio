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
import math
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


# ---------------------------------------------------------------------------
# Production scorer: a faithful port of the live endpoint (irt.py).
#
# It differs from eap_theta above in every respect that matters, so the two are
# kept separate rather than parameterised:
#   * MAP (posterior mode) on a fixed 801-point grid over [-4, 4], step 0.01,
#     not EAP on a 61-point [-6, 6] quadrature.
#   * D = 1.702 -- the metric the reference a values are reported on.
#   * All-correct short-circuits to theta = +4.0; all-incorrect to -4.0.
#   * Unanswered items count as WRONG. The endpoint itself would skip them, but
#     its only caller converts every non-"1" cell to 0 before posting, so blanks
#     never reach it as missing. Matching production means folding that in here.
# Verified against 351 live sessions: 351/351 English, 350/351 Math.
#
# On an adaptive form the website posts only the module-2 variant the student was
# routed to, so items from the other variant never reach the endpoint at all.
# Pass `administered` to reproduce that: un-administered items are excluded from
# the likelihood and from the perfect/zero short-circuit regardless of
# `blanks_wrong`, which governs genuine omits only.
# ---------------------------------------------------------------------------
IRT_D = 1.702
IRT_LO, IRT_HI, IRT_STEP = -4.0, 4.0, 0.01
IRT_N = 801
_PI = 3.141529          # the literal used by the endpoint; kept for fidelity


def _irt_grid():
    theta = [IRT_LO + i * IRT_STEP for i in range(IRT_N)]
    prior = np.array([math.exp(-(t * t / 2.0)) / math.sqrt(2 * _PI) for t in theta])
    return np.array(theta), prior


def irt_theta(responses, a, b, c, blanks_wrong=True, administered=None):
    """MAP ability on the endpoint's grid. responses (N,J), NaN = no response.

    administered (N,J) bool marks the items each student was actually presented;
    None means all of them (a fixed form). Items not administered are missing by
    design -- they never enter the likelihood and never count toward a perfect or
    zero paper.

    blanks_wrong=True reproduces production for the items a student DID see
    (blank -> incorrect). False leaves those omits out of the likelihood too.
    Returns dict with theta, n_seen, n_answered, n_correct, n_blank."""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    R = np.asarray(responses, float)
    n, J = R.shape
    adm = np.ones((n, J), bool) if administered is None else np.asarray(administered, bool)
    no_resp = np.isnan(R)
    blank = no_resp & adm                               # presented, left blank
    X = np.where(no_resp, 0.0, R) if blanks_wrong else R

    theta, prior = _irt_grid()
    post = np.tile(prior, (n, 1))                       # (N,Q)
    # Multiply item by item, in the endpoint's order, so the arithmetic matches.
    for j in range(J):
        z = IRT_D * a[j] * (theta - b[j])
        e = np.exp(z)
        p = c[j] + (1 - c[j]) * (e / (1 + e))           # (Q,)
        x = X[:, j][:, None]
        mult = np.where(x == 1, p[None, :], 1.0 - p[None, :])
        # never administered is always skipped; blanks only when asked
        skip = ~adm[:, j] if blanks_wrong else (~adm[:, j] | blank[:, j])
        mult = np.where(skip[:, None], 1.0, mult)
        post *= mult

    th = theta[np.argmax(post, axis=1)]                 # first max, as the loop does

    # The endpoint's raw_score short-circuit, evaluated over the items the student
    # was presented: raw == 0 means nothing right, raw == n_seen nothing wrong.
    n_corr = np.where(no_resp, 0.0, R).sum(axis=1)
    n_blank = blank.sum(axis=1)
    n_seen = adm.sum(axis=1)
    raw = n_corr if blanks_wrong else n_corr + n_blank
    th = np.where(raw == 0, IRT_LO, th)
    th = np.where((n_seen > 0) & (raw == n_seen), IRT_HI, th)
    return dict(theta=th, n_seen=n_seen.astype(int),
                n_answered=(n_seen - n_blank).astype(int),
                n_correct=n_corr.astype(int), n_blank=n_blank.astype(int))


def to_scaled_irt(theta, mean, sd, floor=None, cap=800.0):
    """theta -> section score exactly as the endpoint does: round to the nearest
    10, cap at `cap`. The endpoint applies no lower bound; pass floor=200 to add
    one (recommended -- without it a weak paper can score below the SAT floor)."""
    v = np.round(np.asarray(theta, float) * sd + mean, -1)
    v = np.minimum(v, cap)
    if floor is not None:
        v = np.maximum(v, floor)
    return v


def score_subject_irt(responses, a, b, c, mean, sd, floor=None, blanks_wrong=True,
                      administered=None):
    """EAP-free scoring path used by the app: MAP theta + endpoint scaling."""
    r = irt_theta(responses, a, b, c, blanks_wrong=blanks_wrong,
                  administered=administered)
    r['scaled'] = to_scaled_irt(r['theta'], mean, sd, floor=floor)
    return r
