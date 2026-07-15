"""
Ability estimation (theta) and scaling to the College Board SAT metric.

Scoring uses EAP (Expected A Posteriori) -- the same default as flexMIRT /
mirt -- which is stable for short module-length tests and yields a proper
conditional standard error (CSEM). Once items are calibrated, scoring is
"fixed-parameter": a brand-new student can be scored instantly without
recalibrating the bank.

Scaling: without College Board's proprietary raw->scale tables we use a
transparent, norm-referenced linear map on theta (default mean 500, SD 100 per
section, clipped to 200-800 in 10-point steps -- the real SAT granularity).
`ScaleConfig` exposes hooks so an official conversion table can be dropped in.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .model import prob_3pl


def eap_scores(responses, a, b, c, D=1.0, n_quad=61, quad_range=4.0,
               prior_mean=0.0, prior_sd=1.0):
    """
    EAP theta and CSEM for each student.

    responses : (P, I) with 0/1/NaN
    a,b,c     : (I,) calibrated item parameters
    Returns   : theta (P,), se (P,)
    """
    X = np.asarray(responses, dtype=float)
    P, I = X.shape
    theta = np.linspace(-quad_range, quad_range, n_quad)
    prior = np.exp(-0.5 * ((theta - prior_mean) / prior_sd) ** 2)
    prior /= prior.sum()

    administered = ~np.isnan(X)
    Xz = np.where(administered, X, 0.0)
    A = administered.astype(float)

    Pmat = prob_3pl(theta[:, None], a[None, :], b[None, :], c[None, :], D)
    Pmat = np.clip(Pmat, 1e-9, 1 - 1e-9)
    logP = np.log(Pmat)
    logQ = np.log(1.0 - Pmat)

    LL = (Xz * A) @ logP.T + ((1 - Xz) * A) @ logQ.T      # (P, Q)
    LL += np.log(prior)[None, :]
    LL -= LL.max(axis=1, keepdims=True)
    post = np.exp(LL)
    post /= post.sum(axis=1, keepdims=True)

    eap = post @ theta
    var = post @ (theta ** 2) - eap ** 2
    se = np.sqrt(np.clip(var, 1e-9, None))
    return eap, se


@dataclass
class ScaleConfig:
    """Linear norm-referenced theta -> section-score map."""
    target_mean: float = 500.0
    target_sd: float = 100.0
    lo: int = 200
    hi: int = 800
    step: int = 10
    # if calibration cohort should define the reference, pass ref_mean/ref_sd;
    # otherwise theta is assumed ~N(0,1) from the identification constraint.
    ref_mean: float = 0.0
    ref_sd: float = 1.0

    def to_scale(self, theta):
        z = (np.asarray(theta) - self.ref_mean) / self.ref_sd
        raw = self.target_mean + z * self.target_sd
        snapped = np.round(raw / self.step) * self.step
        return np.clip(snapped, self.lo, self.hi).astype(int)


def scale_section(theta, cfg: ScaleConfig = None):
    cfg = cfg or ScaleConfig()
    return cfg.to_scale(theta)
