"""Run the full pipeline on every reference mock, write outputs, print an
accuracy report versus the gold standard. Usage: python validate_all.py"""
import sys, os, glob, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))
from pipeline import run_mock, N_MIN
from loader import load_mock
import refjson
from report import write_results_xlsx, validation_table

CSV = {os.path.basename(p).split('_-')[1].split('_')[0]: p
       for p in glob.glob('Digital_SAT_Mock_Test_-*.csv')}
OUT = 'output'


def build_gold():
    """ID-aligned reference: (num,subj) -> (J,3) a,b,c in the loader's item order."""
    gold = {}
    for num in refjson.MOCKS:
        if num not in CSV:
            continue
        for subj, sd in load_mock(CSV[num]).items():
            gold[(num, subj)] = refjson.aligned_gold(num, subj, sd.qids)
    return gold


def main():
    GOLD = build_gold()
    os.makedirs(OUT, exist_ok=True)
    pooled = {('rw', p): [[], []] for p in 'abc'}
    pooled.update({('math', p): [[], []] for p in 'abc'})
    pooled_ok = {k: [[], []] for k in pooled}
    print(f"{'mock':6}{'subj':6}{'param':6}{'r':>8}{'rmse':>9}{'mae':>9}{'n':>6}")
    print('-' * 50)
    t0 = time.time()
    for num in sorted(CSV):
        res = run_mock(CSV[num])
        write_results_xlsx(res, os.path.join(OUT, f'mock_{num}_abc.xlsx'), gold=GOLD, mock_id=num)
        for row in validation_table(res, GOLD, num):
            if row.get('param') in ('a', 'b', 'c'):
                print(f"{num:6}{row['subject']:6}{row['param']:6}{row['r']:8.3f}{row['rmse']:9.3f}{row['mae']:9.3f}{row['n']:6d}")
        # accumulate pooled
        for subj in ('rw', 'math'):
            key = (num, subj)
            if subj not in res or key not in GOLD:
                continue
            items = res[subj]['items']; g = GOLD[key]
            if g.shape[0] != len(items):
                continue
            nadm = np.array([it['n'] for it in items]); ok = nadm >= N_MIN
            for pi, p in enumerate('abc'):
                e = np.array([it[p] for it in items])
                pooled[(subj, p)][0].append(e); pooled[(subj, p)][1].append(g[:, pi])
                pooled_ok[(subj, p)][0].append(e[ok]); pooled_ok[(subj, p)][1].append(g[ok, pi])
        print('-' * 50)
    print(f"\nPOOLED accuracy (fit-on-all; see cv.py for honest leave-one-out):")
    for label, pool in [('ALL items', pooled), (f'WELL-ADMIN (n>={N_MIN})', pooled_ok)]:
        print(f"  [{label}]")
        for subj in ('rw', 'math'):
            for p in 'abc':
                e = np.concatenate(pool[(subj, p)][0]); g = np.concatenate(pool[(subj, p)][1])
                r = np.corrcoef(e, g)[0, 1]; rmse = np.sqrt(np.mean((e - g) ** 2))
                print(f"    {subj:4} {p}: r={r:.3f} rmse={rmse:.3f}")
    print(f"\nWrote per-mock workbooks to {OUT}/  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
