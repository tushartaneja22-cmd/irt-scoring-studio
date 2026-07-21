"""
Data loader for Digital SAT mock CSV exports.

CSV layout (two header rows):
  row 0: sparse row, contains the literal "Question Id's" then the raw numeric ids
  row 1: real header. Metadata cols (Id, Name, Email, Session Id, Total Score,
         Test Taken Date, S1..S4 Score) followed by item columns named like
             S1_24667_(STANDARD)(reading_and_writing)
             S2_24902_(ADAPTIVE)(reading_and_writing)
             S4_25640_(STANDARD)(math)
  rows 2+: one student per row. Item cells are '1' (correct), '0' (incorrect),
           or '-' (not administered -> adaptive routing). Occasional stray
           tokens (e.g. the letter 'O') are treated as missing.

The Digital SAT structure per subject: Module 1 (STANDARD, everyone) + Module 2
(ADAPTIVE, split into a HARD variant and an EASY variant; each student takes
exactly one). We calibrate all items of a subject jointly on one latent trait
(concurrent calibration); adaptive routing is missing-by-design and ignorable
under marginal maximum likelihood.
"""
import csv
import re
from dataclasses import dataclass, field
import numpy as np

ITEM_RE = re.compile(r'^S(\d+)_(\d+)_\((STANDARD|ADAPTIVE)\)\((reading_and_writing|math)\)')


@dataclass
class ItemMeta:
    col: str          # full column name
    section: str      # S1, S2, ...
    qid: str          # question id
    itemtype: str     # STANDARD | ADAPTIVE
    subject: str      # rw | math


@dataclass
class SubjectData:
    subject: str                       # 'rw' or 'math'
    items: list                        # list[ItemMeta] in column order
    responses: np.ndarray              # (n_students, n_items) float, NaN = not administered
    qids: list = field(default_factory=list)

    @property
    def n_students(self):
        return self.responses.shape[0]

    @property
    def n_items(self):
        return self.responses.shape[1]


def _parse_cell(v):
    v = v.strip()
    if v == '1':
        return 1.0
    if v == '0':
        return 0.0
    return np.nan  # '-', '', or stray tokens


def load_mock(path):
    """Load a mock CSV. Returns dict subject -> SubjectData."""
    with open(path, encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.reader(fh))
    header = rows[1]
    item_cols = []
    for i, name in enumerate(header):
        m = ITEM_RE.match(name)
        if m:
            item_cols.append((i, ItemMeta(
                col=name, section='S' + m.group(1), qid=m.group(2),
                itemtype=m.group(3), subject='math' if m.group(4) == 'math' else 'rw')))
    data_rows = rows[2:]
    # keep only rows that look like students (numeric Id in col 0)
    data_rows = [r for r in data_rows if r and r[0].strip().isdigit()]
    n = len(data_rows)

    out = {}
    for subj in ('rw', 'math'):
        cols = [(i, meta) for i, meta in item_cols if meta.subject == subj]
        if not cols:
            continue
        metas = [meta for _, meta in cols]
        M = np.full((n, len(cols)), np.nan)
        for r, row in enumerate(data_rows):
            for j, (i, _) in enumerate(cols):
                if i < len(row):
                    M[r, j] = _parse_cell(row[i])
        out[subj] = SubjectData(subject=subj, items=metas, responses=M,
                                qids=[m.qid for m in metas])
    return out


def load_all(paths):
    return {p: load_mock(p) for p in paths}
