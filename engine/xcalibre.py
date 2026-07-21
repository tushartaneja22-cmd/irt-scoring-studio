"""
xCalibre-faithful 3PL calibration.

Reproduces the estimator behind the reference workbook (Assessment Systems'
xCalibre) rather than approximating it with a learned link:

  * Normal-ogive metric: item response function uses the logistic with the
    scaling constant D = 1.702, and discrimination `a` is reported on that
    metric (a_reported = a_logistic / 1.702) -- this is why xCalibre's `a`
    values sit near 0.5-1.0 rather than ~1.3.
  * Floating priors: Bayesian priors on the item parameters are re-estimated
    from the current pool of item estimates every EM cycle (empirical Bayes),
      log(a) ~ N(mean log a, sd log a)      [floating, sd floored]
      b       ~ N(mean b, sd b)             [floating, sd floored]
    which shrinks the parameters toward the item pool and stabilises small,
    unevenly-administered samples.
  * Fixed guessing prior tied to the number of options:
      logit(c) ~ N(logit c0, sd_g),  c0 = 1/(#options) = 0.25 default.

Estimation is marginal maximum likelihood via EM (Bock-Aitkin) on a fixed
N(0,1) trait, identical machinery to calibrate.py's E-step.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from calibrate import _prep, _quad, _estep, CalibResult

D_NORMAL = 1.702


@dataclass
class XCalConfig:
    D: float = D_NORMAL
    c0: float = 0.25          # guessing prior centre = 1/#options
    sd_g: float = 0.25        # guessing prior sd (logit), fixed
    a_sd_floor: float = 0.12  # floor on floating log-a prior sd
    b_sd_floor: float = 0.5   # floor on floating b prior sd
    a_bounds: tuple = (0.05, 3.0)     # reported metric
    b_bounds: tuple = (-6.0, 6.0)
    c_bounds: tuple = (0.005, 0.5)


def _to_internal(a_rep, b, D):
    """reported normal-ogive (a,b) -> logistic slope/intercept for the E-step."""
    a_int = D * a_rep
    d_int = -D * a_rep * b
    return a_int, d_int


def _mstep_item(Njq, Rjq, theta, cfg, mu_a, sd_a, mu_b, sd_b, logit_c0,
                a0, b0, c0):
    """Maximise expected complete-data loglik + floating priors for one item.
    Params: la=log(a_reported), b, u=logit(c)."""
    D = cfg.D
    lo_c, hi_c = cfg.c_bounds
    u_lo, u_hi = logit(lo_c), logit(hi_c)
    la0 = np.log(min(max(a0, cfg.a_bounds[0] + 1e-3), cfg.a_bounds[1] - 1e-3))
    u0 = logit(min(max(c0, lo_c + 1e-4), hi_c - 1e-4))

    def negobj(params):
        la, b, u = params
        a = np.exp(la)
        g = expit(u)
        z = D * a * (theta - b)
        s = expit(z)
        P = np.clip(g + (1 - g) * s, 1e-9, 1 - 1e-9)
        ll = np.sum(Rjq * np.log(P) + (Njq - Rjq) * np.log1p(-P))
        dLdP = Rjq / P - (Njq - Rjq) / (1 - P)
        sm = s * (1 - s)
        dP_dla = (1 - g) * sm * (D * a * (theta - b))      # dz/dla = z
        dP_db = (1 - g) * sm * (-D * a)
        dP_du = (1 - s) * (g * (1 - g))
        gla = np.sum(dLdP * dP_dla)
        gb = np.sum(dLdP * dP_db)
        gu = np.sum(dLdP * dP_du)
        # floating / fixed priors
        ll += -0.5 * ((la - mu_a) / sd_a) ** 2
        gla += -(la - mu_a) / sd_a ** 2
        ll += -0.5 * ((b - mu_b) / sd_b) ** 2
        gb += -(b - mu_b) / sd_b ** 2
        ll += -0.5 * ((u - logit_c0) / cfg.sd_g) ** 2
        gu += -(u - logit_c0) / cfg.sd_g ** 2
        return -ll, -np.array([gla, gb, gu])

    res = minimize(negobj, np.array([la0, b0, u0]), jac=True, method='L-BFGS-B',
                   bounds=[(np.log(cfg.a_bounds[0]), np.log(cfg.a_bounds[1])),
                           cfg.b_bounds, (u_lo, u_hi)], options={'maxiter': 60})
    la, b, u = res.x
    return float(np.exp(la)), float(b), float(expit(u))


def calibrate_xcalibre(X, cfg=None, n_points=61, max_iter=150, tol=1e-4, verbose=False):
    if cfg is None:
        cfg = XCalConfig()
    Xf, A = _prep(X)
    J = X.shape[1]
    theta, w = _quad(n_points)
    logit_c0 = logit(cfg.c0)

    # init from classical stats (reported metric)
    nadm = A.sum(0)
    p = np.clip(Xf.sum(0) / np.maximum(nadm, 1), 0.05, 0.95)
    a = np.full(J, 0.8)
    pc = np.clip((p - cfg.c0) / (1 - cfg.c0), 0.02, 0.98)
    b = -logit(pc) / (cfg.D * 0.8)                      # rough difficulty
    b = np.clip(b, *cfg.b_bounds)
    c = np.full(J, cfg.c0)

    prev = np.concatenate([a, b, c])
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        # floating prior hyperparameters from current estimates
        la = np.log(np.clip(a, *cfg.a_bounds))
        mu_a, sd_a = float(la.mean()), float(max(la.std(), cfg.a_sd_floor))
        mu_b, sd_b = float(b.mean()), float(max(b.std(), cfg.b_sd_floor))
        # E-step in internal logistic metric
        a_int, d_int = _to_internal(a, b, cfg.D)
        Njq, Rjq = _estep(Xf, A, a_int, d_int, c, theta, w)
        # M-step per item
        for j in range(J):
            a[j], b[j], c[j] = _mstep_item(Njq[j], Rjq[j], theta, cfg,
                                           mu_a, sd_a, mu_b, sd_b, logit_c0,
                                           a[j], b[j], c[j])
        cur = np.concatenate([a, b, c])
        delta = float(np.max(np.abs(cur - prev)))
        prev = cur
        if verbose and (it % 10 == 0 or delta < tol):
            print(f"  xcal iter {it:3d} max|d|={delta:.5f} muA={np.exp(mu_a):.3f} sdA={sd_a:.3f}")
        if delta < tol:
            converged = True
            break

    a_int, d_int = _to_internal(a, b, cfg.D)
    return CalibResult(a=a.copy(), b=b.copy(), c=c.copy(), d=d_int,
                       n_iter=it, converged=converged, loglik=float('nan'))
