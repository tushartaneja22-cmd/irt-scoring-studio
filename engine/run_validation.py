"""Calibrate all 6 mocks and report agreement with the gold standard."""
import glob, os, time, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_mock
from calibrate import calibrate_3pl, PriorConfig
from gold import load_gold, compare, fmt

CSV = {os.path.basename(p).split('_-')[1].split('_')[0]: p
       for p in glob.glob('Digital_SAT_Mock_Test_-*.csv')}


def run(prior=None, n_points=61, subjects=('rw', 'math'), mocks=None):
    gold = load_gold()
    mocks = mocks or sorted(CSV)
    all_e = {'a': [], 'b': [], 'c': []}
    all_g = {'a': [], 'b': [], 'c': []}
    t0 = time.time()
    for num in mocks:
        data = load_mock(CSV[num])
        for subj in subjects:
            if subj not in data:
                continue
            sd = data[subj]
            res = calibrate_3pl(sd.responses, prior=prior, n_points=n_points)
            garr = gold[(num, subj)]
            if garr.shape[0] != sd.n_items:
                print(f"  !! {num} {subj}: est {sd.n_items} vs gold {garr.shape[0]}")
                continue
            m = compare(res.a, res.b, res.c, garr)
            print(fmt(f"{num} {subj:4s} n={sd.n_students} J={sd.n_items} it={res.n_iter}", m))
            all_e['a'].append(res.a); all_g['a'].append(garr[:, 0])
            all_e['b'].append(res.b); all_g['b'].append(garr[:, 1])
            all_e['c'].append(res.c); all_g['c'].append(garr[:, 2])
    print('-' * 100)
    for k in ('a', 'b', 'c'):
        e = np.concatenate(all_e[k]); g = np.concatenate(all_g[k])
        r = np.corrcoef(e, g)[0, 1]
        rmse = np.sqrt(np.mean((e - g) ** 2))
        mae = np.mean(np.abs(e - g))
        print(f"OVERALL {k}: r={r:.4f} rmse={rmse:.4f} mae={mae:.4f}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == '__main__':
    run()
