"""
3PL IRT calibration via Marginal Maximum Likelihood with the EM algorithm
(Bock & Aitkin, 1981), matched to the operational Digital-SAT convention of a
guessing prior centred at 1/(#options) = 0.25.

Model (slope-intercept form, as in mirt):
    P_j(theta) = g_j + (1 - g_j) * sigmoid(a_j * theta + d_j)
    b_j = -d_j / a_j          (returned difficulty)

Latent trait theta ~ N(0, 1), integrated on a fixed Gauss-Hermite-style grid.
Priors (all optional, tuned to reproduce the reference calibration):
    log(a_j) ~ N(mu_a,  sd_a)        lognormal slope prior
    logit(g_j) ~ N(logit(0.25), sd_g) guessing prior centred at 0.25
    d_j ~ N(0, sd_d)                  weak intercept ridge

Concurrent calibration: all items of one subject are estimated jointly on a
single theta; adaptive module routing is missing-by-design and ignorable.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


@dataclass
class PriorConfig:
    # guessing prior (logit-normal, centred at c0)
    c0: float = 0.25
    sd_g: float = 0.5
    use_g_prior: bool = True
    # slope prior (lognormal)
    mu_a: float = 0.0
    sd_a: float = 0.6
    use_a_prior: bool = True
    # intercept ridge
    sd_d: float = 4.0
    use_d_prior: bool = True
    # bounds
    a_bounds: tuple = (0.05, 5.0)
    d_bounds: tuple = (-12.0, 12.0)
    g_bounds: tuple = (0.005, 0.55)


@dataclass
class CalibResult:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    n_iter: int
    converged: bool
    loglik: float


def _quad(n_points=61, lo=-6.0, hi=6.0):
    theta = np.linspace(lo, hi, n_points)
    w = np.exp(-0.5 * theta ** 2)
    w /= w.sum()
    return theta, w


def _prep(X):
    """X: (N,J) with NaN missing. Returns Xf (0-filled), A (admin mask 0/1)."""
    A = (~np.isnan(X)).astype(float)
    Xf = np.where(np.isnan(X), 0.0, X)
    return Xf, A


def _item_probs(a, d, g, theta):
    """Return P (J,Q) and s=sigmoid(z) (J,Q)."""
    z = np.outer(a, theta) + d[:, None]
    s = expit(z)
    P = g[:, None] + (1.0 - g[:, None]) * s
    return P, s


def _estep(Xf, A, a, d, g, theta, w):
    P, _ = _item_probs(a, d, g, theta)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    logP = np.log(P)
    log1mP = np.log1p(-P)
    # person log-likelihood at each node: (N,Q)
    LL = Xf @ logP + (A - Xf) @ log1mP
    LL -= LL.max(axis=1, keepdims=True)
    post = np.exp(LL) * w[None, :]
    denom = post.sum(axis=1, keepdims=True)
    post /= denom
    # expected counts (J,Q)
    Njq = A.T @ post
    Rjq = Xf.T @ post
    # marginal loglik
    ll = np.sum(np.log((np.exp(LL) * w[None, :]).sum(axis=1)) + LL.max(axis=1) * 0)
    return Njq, Rjq


def _mstep_item(Njq, Rjq, theta, prior, a0, d0, g0):
    """Maximize expected complete-data loglik + log prior for one item.
    Params optimized: a (direct), d (direct), u=logit(g)."""
    lo_g, hi_g = prior.g_bounds
    u_lo, u_hi = logit(lo_g), logit(hi_g)
    u0 = logit(min(max(g0, lo_g + 1e-4), hi_g - 1e-4))
    logit_c0 = logit(prior.c0)

    def negobj(params):
        a, d, u = params
        g = expit(u)
        z = a * theta + d
        s = expit(z)
        P = g + (1 - g) * s
        P = np.clip(P, 1e-9, 1 - 1e-9)
        ll = np.sum(Rjq * np.log(P) + (Njq - Rjq) * np.log1p(-P))
        # dL/dP
        dP = Rjq / P - (Njq - Rjq) / (1 - P)
        one_ms = s * (1 - s)
        dPda = (1 - g) * one_ms * theta
        dPdd = (1 - g) * one_ms
        dPdg = 1 - s
        ga = np.sum(dP * dPda)
        gd = np.sum(dP * dPdd)
        gg = np.sum(dP * dPdg)
        gu = gg * (g * (1 - g))  # chain rule g=expit(u)
        # priors (subtract from ll -> objective to maximize)
        if prior.use_a_prior:
            la = np.log(a)
            ll += -0.5 * ((la - prior.mu_a) / prior.sd_a) ** 2
            ga += -((la - prior.mu_a) / prior.sd_a ** 2) * (1.0 / a)
        if prior.use_d_prior:
            ll += -0.5 * (d / prior.sd_d) ** 2
            gd += -d / prior.sd_d ** 2
        if prior.use_g_prior:
            ll += -0.5 * ((u - logit_c0) / prior.sd_g) ** 2
            gu += -(u - logit_c0) / prior.sd_g ** 2
        return -ll, -np.array([ga, gd, gu])

    res = minimize(negobj, np.array([a0, d0, u0]), jac=True, method='L-BFGS-B',
                   bounds=[prior.a_bounds, prior.d_bounds, (u_lo, u_hi)],
                   options={'maxiter': 60})
    a, d, u = res.x
    return a, d, expit(u)


def calibrate_3pl(X, prior=None, n_points=61, max_iter=200, tol=1e-4,
                  init=None, verbose=False):
    """Calibrate 3PL params for response matrix X (N,J), NaN = not administered."""
    if prior is None:
        prior = PriorConfig()
    Xf, A = _prep(X)
    J = X.shape[1]
    theta, w = _quad(n_points)

    # initialization from classical stats
    nadm = A.sum(axis=0)
    p = np.divide(Xf.sum(axis=0), np.maximum(nadm, 1))
    p = np.clip(p, 0.05, 0.95)
    if init is None:
        a = np.full(J, 1.0)
        # d from p ignoring guessing: b0 ~ -logit((p-c)/(1-c)); d=-a*b
        pc = np.clip((p - prior.c0) / (1 - prior.c0), 0.02, 0.98)
        b0 = -logit(pc)
        d = -a * b0
        g = np.full(J, prior.c0)
    else:
        a, d, g = init

    prev = np.concatenate([a, d, g])
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Njq, Rjq = _estep(Xf, A, a, d, g, theta, w)
        for j in range(J):
            a[j], d[j], g[j] = _mstep_item(Njq[j], Rjq[j], theta, prior,
                                           a[j], d[j], g[j])
        cur = np.concatenate([a, d, g])
        delta = np.max(np.abs(cur - prev))
        prev = cur
        if verbose and (it % 10 == 0 or delta < tol):
            print(f"  iter {it:3d}  max|delta|={delta:.5f}")
        if delta < tol:
            converged = True
            break

    b = -d / a
    # final marginal loglik
    P, _ = _item_probs(a, d, g, theta)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    LL = Xf @ np.log(P) + (A - Xf) @ np.log1p(-P)
    m = LL.max(axis=1, keepdims=True)
    ll = float(np.sum(np.log((np.exp(LL - m) * w[None, :]).sum(axis=1)) + m.ravel()))
    return CalibResult(a=a.copy(), b=b, c=g.copy(), d=d.copy(),
                       n_iter=it, converged=converged, loglik=ll)
