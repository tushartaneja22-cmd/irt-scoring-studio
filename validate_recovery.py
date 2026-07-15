"""
Parameter-recovery validation for the MMLE-EM 3PL engine.

Generates responses from KNOWN item parameters and abilities, calibrates
blind, and reports recovery accuracy. This is the standard proof that a
calibrator is correct (Xcalibre/flexMIRT are validated the same way).

We also inject an adaptive-style routing pattern (a Module-1 anchor taken by
everyone plus two Module-2 forms each seen by ~half the sample) so the test
exercises the exact "missing by design" structure of the real SAT data.
"""
import numpy as np
from irt_engine import MMLECalibrator, Priors, prob_3pl

rng = np.random.default_rng(7)


def simulate(n_persons=1500, n_anchor=27, n_form=22, D=1.0):
    # true item params
    def gen(n):
        a = rng.lognormal(mean=0.0, sigma=0.35, size=n)          # ~1
        b = rng.normal(0.0, 1.0, size=n)
        c = rng.uniform(0.10, 0.30, size=n)
        return a, b, c
    aA, bA, cA = gen(n_anchor)          # Module 1 anchor
    aE, bE, cE = gen(n_form); bE -= 0.7  # easy Module-2 form
    aH, bH, cH = gen(n_form); bH += 0.7  # hard Module-2 form
    a = np.concatenate([aA, aE, aH]); b = np.concatenate([bA, bE, bH])
    c = np.concatenate([cA, cE, cH])
    I = len(a)

    theta = rng.normal(0, 1, n_persons)
    P = prob_3pl(theta[:, None], a[None, :], b[None, :], c[None, :], D)
    X = (rng.random((n_persons, I)) < P).astype(float)

    # routing: everyone takes anchor; route to easy/hard M2 by anchor score
    R = np.full_like(X, np.nan)
    R[:, :n_anchor] = X[:, :n_anchor]
    anchor_score = np.nansum(X[:, :n_anchor], axis=1)
    med = np.median(anchor_score)
    hi = anchor_score >= med
    # high performers -> hard form, low -> easy form
    R[hi, n_anchor + n_form:] = X[hi, n_anchor + n_form:]
    R[~hi, n_anchor:n_anchor + n_form] = X[~hi, n_anchor:n_anchor + n_form]
    return R, theta, a, b, c, n_anchor


def main():
    R, theta_true, a, b, c, n_anchor = simulate()
    n_adm = (~np.isnan(R)).sum(axis=0)
    print(f"persons={R.shape[0]} items={R.shape[1]} "
          f"per-item N: min={n_adm.min()} max={n_adm.max()}")

    cal = MMLECalibrator(n_quad=41, D=1.0, max_iter=300, tol=5e-4,
                         priors=Priors(), empirical_bayes=True, verbose=False)
    res = cal.fit(R)

    def report(name, true, est):
        r = np.corrcoef(true, est)[0, 1]
        rmse = np.sqrt(np.mean((true - est) ** 2))
        bias = np.mean(est - true)
        print(f"  {name}: r={r:.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}")

    print(f"\nEM iterations: {res.n_iter}   logL={res.log_marginal:.1f}")
    print("Item parameter recovery (true vs estimated):")
    report("a (discrimination)", a, res.a)
    report("b (difficulty)    ", b, res.b)
    report("c (guessing)      ", c, res.c)

    # ability recovery via EAP
    from irt_engine import eap_scores
    th, se = eap_scores(R, res.a, res.b, res.c, D=res.D)
    report("theta (ability)   ", theta_true, th)

    # difficulty recovery on the low-N routed forms specifically
    lowN = n_adm < np.median(n_adm)
    rb = np.corrcoef(b[lowN], res.b[lowN])[0, 1]
    print(f"\n  b recovery on low-N routed items only: r={rb:.3f} "
          f"(n={lowN.sum()})")
    print("\nPASS thresholds (typical for well-specified 3PL @ N~1500):")
    print("  b r>0.95, a r>0.85, theta r>0.90")


if __name__ == "__main__":
    main()
