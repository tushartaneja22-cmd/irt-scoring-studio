"""Presentation tables built from a `pipeline.run_mock` result.

Pure pandas -- no Streamlit -- so the app and any script share one definition of
what the deliverables look like, and they can be tested directly.
"""
import io
import numpy as np
import pandas as pd

SUBJ_LABEL = {'rw': 'Reading & Writing', 'math': 'Math'}
SHORT = {'rw': 'RW', 'math': 'Math'}
INCOMPLETE_BLANKS = 5          # blanks in one section before a paper is flagged


def items_df(res, subjects):
    """One row per item: ids, classical stats, and the a/b/c actually used."""
    rows = []
    for s in subjects:
        for it in res[s]['items']:
            rows.append({'question_id': it['qid'], 'subject': SUBJ_LABEL[s],
                         'type': it['itemtype'], 'section': it['section'],
                         'n_students': it['n'], 'p_correct': it['p'],
                         'a': it['a'], 'b': it['b'], 'c': it['c'], 'flags': it['flags']})
    return pd.DataFrame(rows)


def scores_df(res, subjects):
    """One row per student: theta, section scores, total, correct + blank counts."""
    base = res[subjects[0]]['scores']
    out = {'Id': base['student_id'], 'Name': base['student_name']}
    total = np.zeros(len(base['student_id']), int)
    for s in subjects:
        sc = res[s]['scores']
        out[f'{SHORT[s]} theta'] = sc['theta']
        out[f'{SHORT[s]} score'] = sc['scaled']
        out[f'{SHORT[s]} correct'] = sc['n_correct']
        out[f'{SHORT[s]} blank'] = sc['n_blank']
        total += np.array(sc['scaled'], int)
    if len(subjects) > 1:
        out['Total'] = total
    df = pd.DataFrame(out)
    blanks = [f'{SHORT[s]} blank' for s in subjects]
    df['Incomplete'] = np.where(df[blanks].max(axis=1) >= INCOMPLETE_BLANKS, 'yes', '')
    return df


def band_df(scores, subj):
    """correct answers -> students, score range, typical score.

    A range rather than a single value is expected: two students with the same
    number correct differ if they got different items right."""
    g = scores.groupby(f'{SHORT[subj]} correct')[f'{SHORT[subj]} score']
    lo, hi = g.min().values, g.max().values
    return pd.DataFrame({
        'correct': g.size().index, 'students': g.size().values,
        'low': lo, 'high': hi,
        'range': [str(a) if a == b else f'{a}-{b}' for a, b in zip(lo, hi)],
        'typical': g.median().round().astype(int).values,
    })


def build_xlsx(items, scores, bands):
    """One workbook: ABC values, Scaled scores, and a band sheet per subject."""
    buf = io.BytesIO()
    sheets = [('ABC values', items), ('Scaled scores', scores)]
    sheets += [(f'{SHORT[s]} band', b) for s, b in bands]
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        for nm, df in sheets:
            df.to_excel(xl, sheet_name=nm, index=False)
            ws = xl.sheets[nm]
            for i, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(300)]) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width, 34)
            ws.freeze_panes = 'A2'
    return buf.getvalue()


def all_tables(res):
    """Everything the app shows, in one call. Returns (subjects, items, scores, bands)."""
    subjects = [s for s in ('rw', 'math') if s in res]
    items = items_df(res, subjects)
    scores = scores_df(res, subjects)
    bands = [(s, band_df(scores, s)) for s in subjects]
    return subjects, items, scores, bands
