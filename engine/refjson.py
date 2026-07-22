"""Load ID-keyed reference item parameters from the per-mock JSON exports (`<id>.txt`).

Each file is the assessment export with a `questions` array; every entry carries the
question id (matching the CSV item qid), `irt_a/irt_b/irt_c`, item `type`
(STANDARD/ADAPTIVE) and a `name` that identifies the subject (Reading and Writing / Math).

Unlike the retired positional Excel, these align to the response matrix by question id,
so every item on every mock can be compared (no dropped-item misalignment).
"""
import json
import os
import numpy as np

MOCKS = ['115', '116', '117', '118', '119', '120']
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(mid, root=None):
    return os.path.join(root or HERE, f'{mid}.txt')


def load_ref_file(mid, root=None):
    """Return list of dicts: qid, a, b, c, type, subject, no."""
    with open(_path(mid, root), encoding='utf-8') as fh:
        d = json.load(fh)
    rows = []
    for q in d['questions']:
        nm = (q.get('name') or '').lower()
        subj = 'math' if 'math' in nm else 'rw'
        rows.append(dict(qid=str(q['question']), a=float(q['irt_a']), b=float(q['irt_b']),
                         c=float(q['irt_c']), type=q.get('type'), subject=subj, no=q.get('no')))
    return rows


def ref_by_qid(mid, root=None):
    """dict qid -> (a, b, c) for one mock (both subjects pooled; qids are unique)."""
    return {r['qid']: (r['a'], r['b'], r['c']) for r in load_ref_file(mid, root)}


def aligned_gold(mid, subject, qids, root=None):
    """Return (J,3) array of reference a,b,c aligned to `qids` order for `subject`.
    Missing qids -> row of NaN (so callers can mask). qids is the loader's item order."""
    ref = ref_by_qid(mid, root)
    out = np.full((len(qids), 3), np.nan)
    for j, qid in enumerate(qids):
        if str(qid) in ref:
            out[j] = ref[str(qid)]
    return out


def load_all_gold(root=None):
    """dict (mid, subject) -> {qid: (a,b,c)} restricted to that subject."""
    gold = {}
    for mid in MOCKS:
        for r in load_ref_file(mid, root):
            gold.setdefault((mid, r['subject']), {})[r['qid']] = (r['a'], r['b'], r['c'])
    return gold
