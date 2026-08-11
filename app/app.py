"""
IRT Scoring Studio — Digital SAT item calibration and scoring.

Upload a mock CSV and get two things:
  1. a / b / c for every item (discrimination / difficulty / guessing)
  2. a scaled score for every student

Scoring reproduces the live endpoint (irt.py) exactly: MAP ability on an
801-point grid over [-4, 4], 3PL with D = 1.702, unanswered items counted
wrong, rounded to the nearest 10 and capped at 800.

Run:  streamlit run app/app.py
"""
import os, sys, io, tempfile
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
from pipeline import run_mock, DEFAULT_PRIOR, N_POINTS, N_MIN, LINK_PATH
from tables import SUBJ_LABEL, SHORT, all_tables, build_xlsx, form_df
import link as linkmod

ROUTING_MIN = 0.90      # below this the ADAPTIVE labels don't match the responses

st.set_page_config(page_title='IRT Scoring Studio', page_icon='📊', layout='wide')
st.markdown("""
<style>
 div[data-testid="stMetricValue"]{font-size:1.3rem}
 .sub{color:#57606a;font-size:0.95rem;margin-top:-0.6rem}
 .note{background:#f6f8fa;border-left:3px solid #d0d7de;padding:0.6rem 0.9rem;
       font-size:0.88rem;color:#424a53;border-radius:0 4px 4px 0}
</style>
""", unsafe_allow_html=True)

st.title('📊 IRT Scoring Studio')
st.markdown('<div class="sub">Upload a Digital SAT mock export → get <b>a, b, c</b> for every '
            'item and a <b>scaled score</b> for every student.</div>', unsafe_allow_html=True)

# ----- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header('Score scale')
    st.caption('Section score = mean + SD × θ. These must match the four fields on '
               'the Assessment record, or the website will report different numbers.')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Reading & Writing**')
        rw_mean = st.number_input('Mean', 200, 800, 670, 10, key='rw_mean')
        rw_sd = st.number_input('SD', 20, 200, 70, 5, key='rw_sd')
    with c2:
        st.markdown('**Math**')
        m_mean = st.number_input('Mean', 200, 800, 690, 10, key='m_mean')
        m_sd = st.number_input('SD', 20, 200, 70, 5, key='m_sd')

    st.divider()
    floor_on = st.toggle('Apply a 200 floor', value=False,
                         help='The live endpoint caps at 800 but has no lower bound, so a '
                              'weak paper can score below 200. Turn this on to clamp; leave '
                              'off to mirror production exactly.')
    blanks_wrong = st.toggle('Unanswered items count as wrong', value=True,
                             help='Production behaviour: the client converts every non-"1" '
                                  'cell to 0 before scoring. Turning this off scores students '
                                  'only on what they attempted, which inflates incomplete papers. '
                                  'This applies to items a student was actually shown — on an '
                                  'adaptive form the module-2 variant they were not routed to is '
                                  'always excluded, either way.')

    with st.expander('Advanced'):
        mode = st.radio('Calibration engine', ['link', 'xcalibre'],
                        format_func=lambda m: {'link': 'Link (best accuracy)',
                                               'xcalibre': 'xCalibre-faithful'}[m])
        n_min = st.slider('Low-confidence threshold (min students per item)', 5, 200, N_MIN, 5)
        n_points = st.select_slider('Quadrature points', [31, 41, 61, 81], N_POINTS)
        a_center = st.number_input('Slope prior centre (a)', 0.3, 2.0,
                                   float(np.exp(DEFAULT_PRIOR['mu_a'])), 0.05)
        sd_a = st.number_input('Slope prior sd (log)', 0.1, 1.0, DEFAULT_PRIOR['sd_a'], 0.05)
        c0 = st.number_input('Guessing prior centre (c)', 0.0, 0.5, DEFAULT_PRIOR['c0'], 0.05)
        sd_g = st.number_input('Guessing prior sd (logit)', 0.1, 1.0, DEFAULT_PRIOR['sd_g'], 0.05)

    prior_cfg = dict(c0=c0, mu_a=float(np.log(a_center)), sd_a=sd_a, sd_g=sd_g,
                     sd_d=DEFAULT_PRIOR['sd_d'])
    score_scale = {
        'rw': dict(mean=float(rw_mean), sd=float(rw_sd),
                   floor=200.0 if floor_on else None, blanks_wrong=blanks_wrong),
        'math': dict(mean=float(m_mean), sd=float(m_sd),
                     floor=200.0 if floor_on else None, blanks_wrong=blanks_wrong),
    }
    st.divider()
    st.caption('Calibration: concurrent 3PL by marginal-maximum-likelihood EM on a fixed '
               'N(0,1) trait, then a learned metric link to the reference scale. '
               'Scoring: MAP on [-4, 4], D = 1.702 — identical to the live endpoint.')


@st.cache_data(show_spinner=False)
def _run(path, prior_cfg, n_points, n_min, mode, score_scale):
    return run_mock(path, prior_cfg=prior_cfg, n_points=n_points, n_min=n_min,
                    mode=mode, score_scale=score_scale)


def guess_mock_id(name):
    import re
    m = re.search(r'-(\d{3})_', name)
    return m.group(1) if m else os.path.splitext(name)[0]


# ----- input ------------------------------------------------------------------
uploads = st.file_uploader('Mock response CSV(s)', type='csv', accept_multiple_files=True)
if not uploads:
    st.info('⬆️ Upload one or more mock CSVs to begin. Each one produces per-item '
            '**a, b, c** and a **scaled score** for every student, downloadable as a '
            'single Excel workbook.')
    st.stop()

for up in uploads:
    mock_id = guess_mock_id(up.name)
    st.markdown(f'## 📄 {up.name}')
    tmp = os.path.join(tempfile.gettempdir(), f'irt_{up.name}')
    with open(tmp, 'wb') as fh:
        fh.write(up.getbuffer())
    with st.spinner(f'Calibrating and scoring {up.name} …'):
        try:
            res = _run(tmp, prior_cfg, int(n_points), int(n_min), mode, score_scale)
        except Exception as e:
            st.error(f'Failed to process {up.name}: {e}')
            continue

    subjects, items, scores, bands = all_tables(res)
    forms = form_df(res, subjects)
    main = 'Total' if 'Total' in scores else f'{SHORT[subjects[0]]} score'

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Students', len(scores))
    k2.metric('Items', len(items))
    k3.metric(f'Mean {main.lower()}', f'{scores[main].mean():.0f}')
    k4.metric('Median', f'{scores[main].median():.0f}')
    k5.metric('Incomplete papers', int((scores['Incomplete'] == 'yes').sum()))

    # ----- how this form was delivered ---------------------------------------
    # Adaptivity is decided per subject: a paper is routinely adaptive in one
    # subject and fixed in the other, so say which is which rather than
    # describing the file as a whole.
    adaptive = [s for s in subjects if res[s]['form']['adaptive']]
    fixed = [s for s in subjects if not res[s]['form']['adaptive']]
    def _phrase(ss):
        return ' and '.join(SUBJ_LABEL[s] for s in ss)
    if adaptive:
        detail = ' · '.join(
            f"{SUBJ_LABEL[s]} {res[s]['form']['form_size']} of "
            f"{res[s]['form']['n_items']} items per student "
            f"({len(res[s]['form']['variants'])} module-2 variants)"
            for s in adaptive)
        st.info(f'**Adaptive: {_phrase(adaptive)}**'
                + (f' · Fixed: {_phrase(fixed)}' if fixed else '')
                + f'  \n{detail}. Items a student was never routed to are excluded '
                  'from scoring rather than counted wrong.')
    else:
        st.success(f'**Fixed form** — every student saw all {len(items)} items '
                   f'({_phrase(subjects)}). No adaptive routing detected.')

    bad = [s for s in adaptive if res[s]['form']['routing'] < ROUTING_MIN]
    if bad:
        st.warning(
            'Routing check failed for ' + _phrase(bad) + ': only '
            + ', '.join(f"{res[s]['form']['routing']:.0%}" for s in bad)
            + ' of students answered inside exactly one module-2 variant. The '
              'export labels these items ADAPTIVE but they were not delivered that '
              'way, so treat the scores as provisional.')

    with st.expander('Form structure'):
        st.dataframe(forms, use_container_width=True, hide_index=True)
        st.caption('`routing check` is the share of students who responded inside '
                   'exactly one module-2 variant — it should be at or near 100% on a '
                   'genuine adaptive form. `students below form size` counts papers '
                   'that ended early.')

    st.download_button('⬇️ Download everything (Excel)',
                       build_xlsx(items, scores, bands, form=forms),
                       file_name=f'mock_{mock_id}_irt.xlsx', key=f'xl_{up.name}',
                       type='primary')

    tabs = st.tabs(['📈 Scaled scores', '📋 Score bands'] +
                   [f'🔢 {SUBJ_LABEL[s]} items' for s in subjects])

    with tabs[0]:
        scale_txt = ' · '.join(f'{SUBJ_LABEL[s]} mean {int(score_scale[s]["mean"])} '
                               f'SD {int(score_scale[s]["sd"])}' for s in subjects)
        st.caption(f'{scale_txt} · rounded to 10 · capped at 800'
                   + (' · floored at 200' if floor_on else ' · no lower bound')
                   + (' · blanks scored wrong' if blanks_wrong else ' · blanks not counted'))
        n_inc = int((scores['Incomplete'] == 'yes').sum())
        if n_inc:
            st.warning(f'{n_inc} student(s) left 5 or more questions blank in a section. '
                       'Their scores are valid but reflect an unfinished paper — see the '
                       '`Incomplete` column.')
        st.dataframe(scores, use_container_width=True, height=420, hide_index=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption(f'{main} distribution')
            st.bar_chart(np.histogram(scores[main], bins=20)[0])
        with c2:
            st.metric('Lowest', f'{scores[main].min():.0f}')
            st.metric('Highest', f'{scores[main].max():.0f}')
            st.download_button('⬇️ Scores (CSV)', scores.to_csv(index=False),
                               file_name=f'mock_{mock_id}_scores.csv',
                               key=f'sc_{up.name}', use_container_width=True)

    with tabs[1]:
        st.caption('What each raw score converted to. A range means students with the same '
                   'number correct scored differently — expected, since which items they got '
                   'right also matters.')
        cols = st.columns(len(bands))
        for col, (s, b) in zip(cols, bands):
            with col:
                f = res[s]['form']
                n_it = (f'{f["form_size"]} of {f["n_items"]} items'
                        if f['adaptive'] else f'{f["n_items"]} items')
                st.markdown(f'**{SUBJ_LABEL[s]}** — {res[s]["n_students"]} students, {n_it}')
                st.dataframe(b[['correct', 'students', 'range', 'typical']],
                             use_container_width=True, height=460, hide_index=True)

    for tab, s in zip(tabs[2:], subjects):
        with tab:
            df = items[items.subject == SUBJ_LABEL[s]]
            n_flag = int((df['flags'] != '').sum())
            m1, m2, m3 = st.columns(3)
            m1.metric('Items', len(df))
            m2.metric('Mean discrimination a', f"{df['a'].mean():.2f}")
            m3.metric('Flagged (low sample)', n_flag)
            st.dataframe(df.drop(columns='subject'), use_container_width=True,
                         height=380, hide_index=True)
            g1, g2 = st.columns(2)
            with g1:
                st.caption('Difficulty (b) distribution')
                st.bar_chart(np.histogram(df['b'], bins=20)[0])
            with g2:
                st.caption('Discrimination (a) vs difficulty (b)')
                st.scatter_chart(df, x='b', y='a', size='n_students')
    st.divider()
