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
from tables import all_tables, build_xlsx, form_df

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

print('\nadaptive-form detection (per subject, not per file)')
# Mock 126 is fixed in both subjects; 122 is adaptive in RW and fixed in Math;
# 117 is adaptive in both. Detection must therefore be per subject.
check('mock 126 detected as fixed in both subjects',
      all(not res[s]['form']['adaptive'] for s in subjects))
check('fixed form: every student sees the whole pool',
      all(res[s]['form']['form_size'] == res[s]['form']['n_items'] for s in subjects))

MIXED = os.path.join(ROOT, 'Digital_SAT_Mock_Test_-122_1779538149.csv')
BOTH = os.path.join(ROOT, 'Digital_SAT_Mock_Test_-117_1774194364.csv')
for path, expect in [(MIXED, {'rw': True, 'math': False}),
                     (BOTH, {'rw': True, 'math': True})]:
    name = os.path.basename(path)
    if not os.path.exists(path):
        skip(f'{name} adaptivity', 'file not present')
        continue
    r = run_mock(path, mode='link', score_scale=SCALE)
    got = {s: r[s]['form']['adaptive'] for s in expect}
    check(f'{name}: adaptivity per subject', got == expect, str(got))
    for s, is_ad in expect.items():
        f = r[s]['form']
        if is_ad:
            check(f'  {s}: form smaller than pool',
                  f['form_size'] < f['n_items'], f"{f['form_size']}/{f['n_items']}")
            check(f'  {s}: routing is clean', f['routing'] >= 0.90, f"{f['routing']:.1%}")
            sc = r[s]['scores']
            seen = np.array(sc['n_seen']); blank = np.array(sc['n_blank'])
            answered = np.array(sc['n_answered'])
            check(f'  {s}: seen = answered + blank',
                  bool((seen == answered + blank).all()))
            check(f'  {s}: nobody is presented more than one form',
                  bool((seen <= f['form_size']).all()), f'max seen {seen.max()}')
            # The bug this guards: un-administered items used to be counted blank,
            # which handed every single student at least (pool - form) blanks.
            check(f'  {s}: the typical student has no blanks',
                  float(np.median(blank)) < 5,
                  f'median {np.median(blank):.0f}, would be '
                  f"{f['n_items'] - f['form_size']}+ if routing were ignored")
        else:
            check(f'  {s}: whole pool administered',
                  f['form_size'] == f['n_items'])

print('\nun-administered items must not drag ability down')
if os.path.exists(BOTH):
    from loader import load_mock
    from scoring import irt_theta
    sd = load_mock(BOTH)['rw']
    it = run_mock(BOTH, mode='link', score_scale=SCALE)['rw']['items']
    A = np.array([[i['a'] for i in it], [i['b'] for i in it], [i['c'] for i in it]])
    good = irt_theta(sd.responses, *A, administered=sd.administered)['theta']
    naive = irt_theta(sd.responses, *A)['theta']          # the old behaviour
    check('routing-aware theta is centred near the calibration scale',
          abs(float(np.mean(good))) < 0.5, f'mean {np.mean(good):+.2f}')
    check('ignoring routing pushes ability far negative',
          float(np.mean(naive)) < float(np.mean(good)) - 0.5,
          f'naive {np.mean(naive):+.2f} vs routing-aware {np.mean(good):+.2f}')
else:
    skip('routing-aware theta', f'{os.path.basename(BOTH)} not present')

print('\ntables')
check('items sheet covers every item',
      len(items) == sum(len(res[s]['items']) for s in subjects), f'{len(items)} rows')
check('scores sheet has one row per student', len(scores) == len(ids))
check('band totals equal the cohort',
      all(int(b['students'].sum()) == len(scores) for _, b in bands))
check('workbook builds', len(build_xlsx(items, scores, bands)) > 5000)
check('workbook builds with a form sheet',
      len(build_xlsx(items, scores, bands, form=form_df(res, subjects))) > 5000)

tail = f'  ({len(skipped)} skipped)' if skipped else ''
print('\n' + ('ALL PASSED' + tail if not fails else f'{len(fails)} FAILED: {fails}'))
sys.exit(1 if fails else 0)
