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

Adaptivity is decided PER SUBJECT, not per file: a form is routinely adaptive in
Reading & Writing and fixed in Math, or the reverse. A subject is adaptive when
its items carry the ADAPTIVE type across two or more sections (the module-2
variants); `routing` on the returned SubjectData reports how cleanly students
actually split between them, so a mislabelled export is visible rather than
silently mis-scored.

Two different meanings hide behind a missing cell, and scoring must tell them
apart:
  * not administered -- the module-2 variant this student was never routed to.
    Missing by design; it must drop out of the likelihood entirely. Counting it
    wrong penalises every student for 27 RW / 22 Math items they never saw.
  * omitted -- presented and left blank. Scored WRONG in production.
The '-' token alone cannot distinguish them (a fixed form uses '-' for genuine
omits too), so `administered` is derived from the routing structure instead.
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
    responses: np.ndarray              # (n_students, n_items) float, NaN = no response
    qids: list = field(default_factory=list)
    student_ids: list = field(default_factory=list)   # aligned to responses rows
    student_names: list = field(default_factory=list)
    administered: np.ndarray = None    # (n_students, n_items) bool, True = presented
    variants: list = field(default_factory=list)      # module-2 section names
    routing: float = 1.0               # share of students on exactly one variant

    def __post_init__(self):
        if self.administered is None:
            self.administered = np.ones(self.responses.shape, bool)

    @property
    def n_students(self):
        return self.responses.shape[0]

    @property
    def n_items(self):
        return self.responses.shape[1]

    @property
    def is_adaptive(self):
        """True when students are routed to one of several module-2 variants."""
        return len(self.variants) > 1

    @property
    def n_seen(self):
        """Per student: how many items were actually presented."""
        return self.administered.sum(1)

    @property
    def form_size(self):
        """Items on one student's form -- < n_items exactly when adaptive."""
        if self.n_students == 0:
            return self.n_items
        return int(np.median(self.n_seen))

    @property
    def n_omitted(self):
        """Per student: presented but left blank. These are scored wrong."""
        return (self.administered & np.isnan(self.responses)).sum(1)


def _parse_cell(v):
    v = v.strip()
    if v == '1':
        return 1.0
    if v == '0':
        return 0.0
    return np.nan  # '-', '', or stray tokens


def _routing(metas, responses):
    """Work out which items each student was actually presented.

    Module-2 variants are the sections carrying ADAPTIVE items; a student takes
    exactly one, so the others are missing by design. The variant a student was
    routed to is the one they responded in. Students who never reached module 2
    (abandoned the paper) count as not administered on every variant rather than
    being scored wrong on all of them.

    Returns (administered bool (N,J), variant section names, routing quality).
    Routing quality is the share of students who responded in exactly one
    variant -- near 1.0 confirms genuine routing, low values mean the ADAPTIVE
    label does not describe how the form was actually delivered."""
    n, J = responses.shape
    sec = np.array([m.section for m in metas])
    variants = sorted({m.section for m in metas if m.itemtype == 'ADAPTIVE'})
    adm = np.ones((n, J), bool)
    if len(variants) < 2 or n == 0:
        return adm, variants, 1.0

    answered = ~np.isnan(responses)
    # (N, V) responses each student gave inside each variant section
    per_variant = np.column_stack([answered[:, sec == v].sum(1) for v in variants])
    taken = per_variant.argmax(1)               # ties -> first, as np.argmax does
    reached = per_variant.max(1) > 0

    for i, v in enumerate(variants):
        cols = sec == v
        adm[np.ix_((taken != i) | ~reached, cols)] = False
    routing = float(((per_variant > 0).sum(1) == 1).mean())
    return adm, variants, routing


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

    # student identifiers (aligned to response rows); Name column is optional
    def _find(name):
        for i, h in enumerate(header):
            if h.strip().lower() == name:
                return i
        return None
    id_i, name_i = 0, _find('name')
    student_ids = [row[id_i].strip() if id_i < len(row) else '' for row in data_rows]
    student_names = [row[name_i].strip() if name_i is not None and name_i < len(row) else ''
                     for row in data_rows]

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
        adm, variants, routing = _routing(metas, M)
        out[subj] = SubjectData(subject=subj, items=metas, responses=M,
                                qids=[m.qid for m in metas],
                                student_ids=student_ids, student_names=student_names,
                                administered=adm, variants=variants, routing=routing)
    return out


def load_all(paths):
    return {p: load_mock(p) for p in paths}
