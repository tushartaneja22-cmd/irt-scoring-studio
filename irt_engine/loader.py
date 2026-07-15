"""
Parser for the Digital SAT adaptive mock-test CSV export.

Layout (per file = one test form):
  row 0 : sparse "Question Id's" header with raw item ids
  row 1 : real column header
          cols 0-9  = Id, Name, Email, Session Id, Total Score,
                      Test Taken Date, S1 Score..S4 Score
          cols 10+  = S{n}_{itemid}_({STANDARD|ADAPTIVE})({subject})
                      cell value 1=correct, 0=incorrect, '-'=not administered
  rows 2+: one student each

Section -> role (College Board 2-stage adaptive):
  S1        RW  Module 1 (anchor, ~all students)
  S2 / S3   RW  Module 2 forms (routed: one or the other)
  S4        Math Module 1 (anchor)
  S5 / S6   Math Module 2 forms (routed)

We split each form into two unidimensional calibration sets: reading_and_writing
and math. The '-' cells become NaN (missing by design), which the calibrator
skips -- the shared Module-1 anchor keeps the routed Module-2 forms on one scale.
"""
from __future__ import annotations

import csv
import os
import numpy as np
from dataclasses import dataclass


@dataclass
class FormData:
    form: str
    subject: str                 # 'reading_and_writing' | 'math'
    student_ids: list
    names: list
    emails: list
    item_ids: list               # raw item ids (str)
    item_sections: list          # S1..S6
    responses: np.ndarray        # (P, I) 0/1/NaN
    meta: dict                   # extra columns kept for reference


def _parse_col(colname):
    """'S1_24667_(STANDARD)(reading_and_writing)' -> (S1, 24667, STANDARD, subj)"""
    section = colname.split("_", 1)[0]
    rest = colname[len(section) + 1:]
    # item id is up to the first '_('
    idpart = rest.split("_(", 1)[0]
    tags = [t.rstrip(")") for t in rest.split("(")[1:]]
    itemtype = tags[0] if tags else ""
    subject = tags[1] if len(tags) > 1 else ""
    return section, idpart, itemtype, subject


def load_form(path):
    """Return {'reading_and_writing': FormData, 'math': FormData} for one CSV."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return _rows_to_forms(rows, os.path.splitext(os.path.basename(path))[0])


def load_form_bytes(data, form_name="uploaded"):
    """Same as load_form but from raw bytes (e.g. a Streamlit upload)."""
    import io
    text = data.decode("utf-8-sig") if isinstance(data, (bytes, bytearray)) else data
    rows = list(csv.reader(io.StringIO(text)))
    return _rows_to_forms(rows, form_name)


def _rows_to_forms(rows, form):
    header = rows[1]
    data = [r for r in rows[2:] if len(r) > 3 and r[0].strip()]

    # Detect item columns by their subject tag, NOT by a fixed position, so the
    # parser is robust to any number of leading meta / section-score columns and
    # to any number of modules (1 non-adaptive, 2, or 3 adaptive forms).
    cols = []
    for j, name in enumerate(header):
        if "(reading_and_writing)" in name or "(math)" in name:
            section, iid, itype, subj = _parse_col(name)
            cols.append((j, section, iid, itype, subj))

    student_ids = [r[0].strip() for r in data]
    names = [r[1].strip() for r in data]
    emails = [r[2].strip().lower() for r in data]

    out = {}
    for subject in ("reading_and_writing", "math"):
        sel = [(j, sec, iid) for (j, sec, iid, itype, subj) in cols
               if subj == subject]
        if not sel:
            continue
        I = len(sel)
        P = len(data)
        R = np.full((P, I), np.nan)
        for ci, (j, sec, iid) in enumerate(sel):
            for pi, r in enumerate(data):
                v = r[j].strip() if j < len(r) else "-"
                if v == "1":
                    R[pi, ci] = 1.0
                elif v == "0":
                    R[pi, ci] = 0.0
                # '-' or blank -> NaN
        out[subject] = FormData(
            form=form, subject=subject,
            student_ids=student_ids, names=names, emails=emails,
            item_ids=[s[2] for s in sel],
            item_sections=[s[1] for s in sel],
            responses=R, meta={},
        )
    return out


def load_folder(folder):
    """Load every CSV in a folder -> list[FormData] (2 per file)."""
    forms = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".csv"):
            for subject, fd in load_form(os.path.join(folder, fn)).items():
                forms.append(fd)
    return forms
