"""
Scale linking / equating.

IRT parameters are identified only up to a linear transform of the theta scale
(theta* = A*theta + B). To compare two calibrations -- ours vs Xcalibre, or two
of our forms sharing common items/persons -- we must first place them on one
metric. We provide the two standard families:

  * mean/mean and mean/sigma moment methods (fast, robust)
  * Stocking-Lord characteristic-curve method (minimises the difference between
    the two test characteristic curves; the field standard, what IRTEQ uses)

Transform applied to the "new" form to bring it onto the "reference" metric:
    b_new*  = A*b_new + B
    a_new*  = a_new / A
    c       unchanged
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .model import prob_3pl


def mean_sigma(a_new, b_new, a_ref, b_ref):
    """A, B from matching the mean & SD of b (uses common items, aligned)."""
    A = np.std(b_ref) / np.std(b_new)
    B = np.mean(b_ref) - A * np.mean(b_new)
    return A, B


def mean_mean(a_new, b_new, a_ref, b_ref):
    """A from a-ratio, B from b-means (Loyd & Hoover)."""
    A = np.mean(a_new) / np.mean(a_ref)
    B = np.mean(b_ref) - A * np.mean(b_new)
    return A, B


def stocking_lord(a_new, b_new, c_new, a_ref, b_ref, c_ref,
                  D=1.702, n_quad=41, quad_range=4.0):
    """
    Stocking-Lord: find (A, B) minimising the squared gap between the two test
    characteristic curves over a theta grid. Common items must be aligned
    row-for-row between *_new and *_ref.
    """
    theta = np.linspace(-quad_range, quad_range, n_quad)
    w = np.exp(-0.5 * theta ** 2); w /= w.sum()
    a_new, b_new, c_new = map(np.asarray, (a_new, b_new, c_new))
    a_ref, b_ref, c_ref = map(np.asarray, (a_ref, b_ref, c_ref))

    def tcc(a, b, c, th):
        # sum over items of P_i(th): (Q,)
        P = prob_3pl(th[:, None], a[None, :], b[None, :], c[None, :], D)
        return P.sum(axis=1)

    ref_curve = tcc(a_ref, b_ref, c_ref, theta)

    def loss(x):
        A, B = x
        a_t = a_new / A
        b_t = A * b_new + B
        new_curve = tcc(a_t, b_t, c_new, theta)
        return np.sum(w * (new_curve - ref_curve) ** 2)

    res = minimize(loss, [1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-8})
    return float(res.x[0]), float(res.x[1])


def apply_transform(a, b, c, A, B):
    """Return (a*, b*, c) placed on the reference metric."""
    return np.asarray(a) / A, A * np.asarray(b) + B, np.asarray(c)
