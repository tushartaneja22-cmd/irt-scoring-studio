"""Batch CLI for IRT Scoring Studio.

Examples:
  python engine/cli.py "Digital_SAT_Mock_Test_-115_1754312309.csv"
  python engine/cli.py data/*.csv --out output --gold "A, B, C Values.xlsx"
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_mock, N_MIN
from report import write_results_xlsx, validation_table
from gold import load_gold


def guess_id(name):
    import re
    m = re.search(r'-(\d{3})_', os.path.basename(name))
    return m.group(1) if m else os.path.splitext(os.path.basename(name))[0]


def main():
    ap = argparse.ArgumentParser(description='Digital SAT 3PL item calibration (a,b,c).')
    ap.add_argument('inputs', nargs='+', help='mock CSV file(s) or globs')
    ap.add_argument('--out', default='output', help='output directory')
    ap.add_argument('--gold', default=None, help='optional gold-standard xlsx for validation')
    ap.add_argument('--n-min', type=int, default=N_MIN, help='low-confidence student threshold')
    ap.add_argument('--mode', choices=['link', 'xcalibre'], default='link',
                    help="engine: 'link' (best RMSE) or 'xcalibre' (xCalibre-faithful)")
    ap.add_argument('--correct', choices=['off', 'bias', 'moment', 'regress', 'exact'],
                    default='off',
                    help="artificial gold-anchored correction (needs --gold; referenced "
                         "mocks only). off|bias|moment|regress|exact")
    ap.add_argument('--strength', type=float, default=1.0,
                    help='correction blend 0..1 (0=off, 1=full mode)')
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(glob.glob(pat) or [pat])
    os.makedirs(args.out, exist_ok=True)
    gold = load_gold(args.gold) if args.gold else None

    for path in paths:
        if not os.path.exists(path):
            print(f'!! not found: {path}'); continue
        mid = guess_id(path)
        t0 = time.time()
        res = run_mock(path, n_min=args.n_min, mode=args.mode, gold=gold, mock_id=mid,
                       correction=args.correct, strength=args.strength)
        if args.correct != 'off' and gold:
            applied = [s for s in res if res[s].get('correction', 'off') != 'off']
            print(f'      correction {args.correct}@{args.strength:.2f} applied to: '
                  f'{", ".join(applied) if applied else "(none — no aligned gold)"}')
        outp = os.path.join(args.out, f'mock_{mid}_abc.xlsx')
        write_results_xlsx(res, outp, gold=gold, mock_id=mid)
        n_items = sum(len(res[s]['items']) for s in res)
        print(f'[{mid}] {n_items} items, {time.time()-t0:.1f}s -> {outp}')
        if gold:
            for r in validation_table(res, gold, mid):
                if r.get('param') in ('a', 'b', 'c'):
                    print(f'      {r["subject"]:4} {r["param"]}: r={r["r"]:.3f} rmse={r["rmse"]:.3f}')


if __name__ == '__main__':
    main()
