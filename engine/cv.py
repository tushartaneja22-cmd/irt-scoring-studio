"""Leave-one-mock-out cross-validation of the calibrator + metric link.

This is the honest estimate of accuracy on a brand-new mock: for each held-out
mock the link is fit on the other mocks (per subject) and applied to the held-out
one, compared to the ID-keyed JSON reference. Delegates to validate_loo, which
runs the production link code path (see engine/validate_loo.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from validate_loo import run

if __name__ == '__main__':
    run()
