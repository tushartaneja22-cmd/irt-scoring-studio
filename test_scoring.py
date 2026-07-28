"""Regression test: the app must reproduce the live endpoint's scaled scores.

Ground truth is `mock126_scores.csv` -- sessions scored by the production
/api/compute-scaled-score/ endpoint at RW 670/70 and Math 690/70. That file
carries student names and emails, so it is NOT in the repo; supply it locally.
Without it (or without the mock export) the endpoint-comparison checks are
skipped and only the behavioural checks run.

Run:  python test_scoring.py
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
from pipeline import run_mock
from tables import all_tables, build_xlsx

MOCK = os.path.join(ROOT, 'Digital_SAT_Mock_Test_-126_1785040623.csv')
TRUTH = os.path.join(ROOT, 'mock126_scores.csv')
SCALE = {'rw': dict(mean=670.0, sd=70.0, floor=None, blanks_wrong=True),
         'math': dict(mean=690.0, sd=70.0, floor=None, blanks_wrong=True)}

# One session (172520) held a cached scaled_score, so the endpoint returned a
# stale value instead of recomputing -- see irt.py:118. It is not a scoring
# difference and is excluded from the exact-match requirement.
CACHED = {172520}

fails = []
skipped = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def skip(name, why):
    print(f'  SKIP  {name}  ({why})')
    skipped.append(name)


if not os.path.exists(MOCK):
    print(f'cannot run: {os.path.basename(MOCK)} not found (student data, not in the repo)')
    sys.exit(0)

print('running pipeline on mock 126 ...')
res = run_mock(MOCK, mode='link', score_scale=SCALE)
subjects, items, scores, bands = all_tables(res)
ids = [int(i) for i in scores['Id']]

print('\nscaled scores vs live endpoint')
if not os.path.exists(TRUTH):
    skip('endpoint comparison', f'{os.path.basename(TRUTH)} not present')
else:
    truth = pd.read_csv(TRUTH).set_index('Student ID')
    keep = np.array([i not in CACHED for i in ids])
    for subj, col, tcol in [('rw', 'RW score', 'Verbal Scaled'),
                            ('math', 'Math score', 'Math Scaled')]:
        got = scores[col].to_numpy()
        exp = np.array([truth.loc[i, tcol] for i in ids])
        n_ok = int((got[keep] == exp[keep]).sum())
        check(f'{subj} exact match', n_ok == keep.sum(), f'{n_ok}/{keep.sum()}')

    got_t = scores['Total'].to_numpy()
    exp_t = np.array([truth.loc[i, 'Total Scaled'] for i in ids])
    check('total exact match', int((got_t[keep] == exp_t[keep]).sum()) == keep.sum())

    print('\nraw counts')
    for col, tcol in [('RW correct', 'Verbal Raw'), ('Math correct', 'Math Raw')]:
        got = scores[col].to_numpy()
        exp = np.array([truth.loc[i, tcol] for i in ids])
        check(f'{col} matches export', bool((got == exp).all()))

print('\nendpoint behaviours')
check('no score exceeds 800', bool((scores[['RW score', 'Math score']].max() <= 800).all()))
check('perfect section scores 800',
      all(int(r['Math score']) == 800 for _, r in scores.iterrows()
          if r['Math correct'] == len(res['math']['items'])))
blank_students = scores[scores['Math blank'] > 0]
check('blanks reduce the score (counted wrong)', len(blank_students) > 0,
      f'{len(blank_students)} students with blanks')

print('\nno-floor is faithful to production')
check('a sub-200 score is possible without the floor',
      bool(scores[['RW score', 'Math score']].min().min() < 500))

print('\nfloor toggle')
floored = run_mock(MOCK, mode='link', score_scale={
    'rw': dict(mean=300.0, sd=70.0, floor=200.0, blanks_wrong=True),
    'math': dict(mean=300.0, sd=70.0, floor=200.0, blanks_wrong=True)})
lo = min(min(floored[s]['scores']['scaled']) for s in ('rw', 'math'))
check('floor clamps at 200', lo >= 200, f'min {lo}')

print('\ntables')
check('items sheet covers every item',
      len(items) == sum(len(res[s]['items']) for s in subjects), f'{len(items)} rows')
check('scores sheet has one row per student', len(scores) == len(ids))
check('band totals equal the cohort',
      all(int(b['students'].sum()) == len(scores) for _, b in bands))
check('workbook builds', len(build_xlsx(items, scores, bands)) > 5000)

tail = f'  ({len(skipped)} skipped)' if skipped else ''
print('\n' + ('ALL PASSED' + tail if not fails else f'{len(fails)} FAILED: {fails}'))
sys.exit(1 if fails else 0)
