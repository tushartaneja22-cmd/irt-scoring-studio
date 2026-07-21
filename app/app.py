"""
IRT Scoring Studio — Digital SAT 3PL item calibration.

Upload a mock CSV (same export format as the reference mocks) and the app
produces a / b / c (discrimination / difficulty / guessing) for every item,
with QC flags and Excel / CSV downloads. Upload CSV, get a/b/c — that's it.

Run:  streamlit run app/app.py   (or streamlit run app.py from the repo root)
"""
import os, sys, io, tempfile
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
from pipeline import run_mock, DEFAULT_PRIOR, N_POINTS, N_MIN, LINK_PATH
from report import write_results_xlsx
import link as linkmod

st.set_page_config(page_title='IRT Scoring Studio', page_icon='📊', layout='wide')

# ----- header -----------------------------------------------------------------
st.markdown("""
<style>
 .metric-good {color:#1a7f37;font-weight:600}
 .metric-warn {color:#b7791f;font-weight:600}
 div[data-testid="stMetricValue"]{font-size:1.35rem}
 .subtitle{color:#57606a;font-size:0.95rem;margin-top:-0.6rem}
</style>
""", unsafe_allow_html=True)
st.title('📊 IRT Scoring Studio')
st.markdown('<div class="subtitle">Digital SAT 3-parameter logistic (3PL) item calibration '
            '— produces <b>a</b> (discrimination), <b>b</b> (difficulty), <b>c</b> (guessing).</div>',
            unsafe_allow_html=True)

# ----- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header('Settings')
    mode = st.radio('Calibration engine', ['link', 'xcalibre'],
                    format_func=lambda m: {'link': 'Link (best accuracy)',
                                           'xcalibre': 'xCalibre-faithful'}[m],
                    help='link: MML-EM + learned metric link (lowest held-out RMSE). '
                         'xcalibre: normal-ogive D=1.702 + floating priors, reproducing '
                         "xCalibre's estimator conventions directly (comparable accuracy, "
                         'slower).')
    n_min = st.slider('Low-confidence threshold (min students per item)', 5, 200, N_MIN, 5,
                      help='Items administered to fewer students than this are flagged '
                           '(common for lightly-taken adaptive modules).')
    with st.expander('Scaled-score scale'):
        st.caption('Section score = mean + sd·θ (θ standardised on N(0,1)), clamped to '
                   '[200, 800] and rounded to 10. Norm-referenced — **not** the official '
                   'College Board raw-to-scaled table.')
        sc_mean = st.number_input('Section mean', 200, 800, 500, 10)
        sc_sd = st.number_input('Section sd', 20, 200, 100, 5)
    score_scale = {'mean': float(sc_mean), 'sd': float(sc_sd)}
    with st.expander('Advanced (model priors)'):
        st.caption('MAP priors that stabilise small-sample estimation. Defaults reproduce '
                   'the reference calibration; change only if you know what you are doing.')
        a_center = st.number_input('Slope prior centre (a)', 0.3, 2.0, float(np.exp(DEFAULT_PRIOR['mu_a'])), 0.05)
        sd_a = st.number_input('Slope prior sd (log)', 0.1, 1.0, DEFAULT_PRIOR['sd_a'], 0.05)
        c0 = st.number_input('Guessing prior centre (c)', 0.0, 0.5, DEFAULT_PRIOR['c0'], 0.05)
        sd_g = st.number_input('Guessing prior sd (logit)', 0.1, 1.0, DEFAULT_PRIOR['sd_g'], 0.05)
        n_points = st.select_slider('Quadrature points', [31, 41, 61, 81], N_POINTS)
    prior_cfg = dict(c0=c0, mu_a=float(np.log(a_center)), sd_a=sd_a, sd_g=sd_g, sd_d=DEFAULT_PRIOR['sd_d'])
    st.divider()
    st.caption('Method: concurrent 3PL calibration by marginal-maximum-likelihood EM '
               '(Bock–Aitkin) on a fixed N(0,1) trait, then a learned metric link to the '
               'reference scale. Adaptive-module routing is missing-by-design and ignorable.')
    model = linkmod.load(LINK_PATH)
    st.caption(f"Link model trained on mocks: {', '.join(model.get('trained_on', []))}")

# ----- inputs -----------------------------------------------------------------
uploads = st.file_uploader('Mock response CSV(s)', type='csv', accept_multiple_files=True)


@st.cache_data(show_spinner=False)
def _run(path, prior_cfg, n_points, n_min, mode, score_scale):
    return run_mock(path, prior_cfg=prior_cfg, n_points=n_points, n_min=n_min, mode=mode,
                    score_scale=score_scale)


def guess_mock_id(name):
    import re
    m = re.search(r'-(\d{3})_', name)
    return m.group(1) if m else name


def color_flag(row):
    return ['background-color:#fce4d6' if row['flags'] else '' for _ in row]


def scaled_scores_df(res, subjects):
    """Combine per-student EAP scaled scores across subjects (rows are aligned to
    the same students). Adds a Total when both sections are present."""
    label = {'rw': 'RW', 'math': 'Math'}
    base = res[subjects[0]]['scores']
    n = len(base['student_id'])
    data = {'Id': base['student_id'], 'Name': base['student_name']}
    total = np.zeros(n)
    for s in subjects:
        sc = res[s]['scores']
        data[f'{label[s]} (200–800)'] = sc['scaled']
        data[f'{label[s]} answered'] = sc['n_answered']
        total += np.asarray(sc['scaled'])
    if len(subjects) > 1:
        data['Total (400–1600)'] = total.astype(int)
    return pd.DataFrame(data)


if not uploads:
    st.info('⬆️ Upload one or more mock CSVs to begin. You get per-item **a, b, c** '
            '(reference-workbook layout) plus a **Scaled scores** tab with each student’s '
            'section and total scores — all downloadable as Excel or CSV.')
    st.stop()

for up in uploads:
    mock_id = guess_mock_id(up.name)
    st.markdown(f'## 📄 {up.name}')
    tmp = os.path.join(tempfile.gettempdir(), f'irt_{up.name}')
    with open(tmp, 'wb') as fh:
        fh.write(up.getbuffer())
    with st.spinner(f'Calibrating {up.name} …'):
        try:
            res = _run(tmp, prior_cfg, int(n_points), int(n_min), mode, score_scale)
        except Exception as e:
            st.error(f'Failed to process {up.name}: {e}')
            continue

    subjects = [s for s in ('rw', 'math') if s in res]
    tab_labels = [{'rw': 'Reading & Writing', 'math': 'Math'}[s] for s in subjects]
    tabs = st.tabs(tab_labels + ['📈 Scaled scores'])
    for tab, subj in zip(tabs, subjects):
        with tab:
            info = res[subj]
            items = info['items']
            df = pd.DataFrame(items)
            n_flag = int((df['flags'] != '').sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Students', info['n_students'])
            m2.metric('Items', len(items))
            m3.metric('Mean discrimination a', f"{df['a'].mean():.2f}")
            m4.metric('Flagged items', n_flag)

            show = df[['qid', 'section', 'itemtype', 'n', 'p', 'a', 'b', 'c', 'flags']]
            st.dataframe(show.style.apply(color_flag, axis=1), use_container_width=True,
                         height=340, hide_index=True)

            # distributions
            g1, g2 = st.columns(2)
            with g1:
                st.caption('Difficulty (b) distribution')
                st.bar_chart(np.histogram(df['b'], bins=20)[0])
            with g2:
                st.caption('Discrimination (a) vs difficulty (b)')
                st.scatter_chart(df, x='b', y='a', size='n')

    # ----- scaled scores tab --------------------------------------------------
    with tabs[-1]:
        ss = scaled_scores_df(res, subjects)
        st.caption(f'Per-student **EAP ability (θ)** from the calibrated 3PL items, mapped to '
                   f'a section scale (mean {int(sc_mean)}, sd {int(sc_sd)}, clamped 200–800, '
                   'rounded to 10). Norm-referenced — not the official College Board conversion.')
        main_col = 'Total (400–1600)' if 'Total (400–1600)' in ss else ss.columns[2]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('Students', len(ss))
        s2.metric(f'Mean {main_col.split(" (")[0].lower()}', f"{ss[main_col].mean():.0f}")
        s3.metric('Median', f"{ss[main_col].median():.0f}")
        s4.metric('Min / Max', f"{ss[main_col].min():.0f} / {ss[main_col].max():.0f}")
        st.dataframe(ss, use_container_width=True, height=380, hide_index=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption(f'{main_col} distribution')
            st.bar_chart(np.histogram(ss[main_col], bins=20)[0])
        with c2:
            st.download_button('⬇️ Download scaled scores (CSV)', ss.to_csv(index=False),
                               file_name=f'mock_{mock_id}_scaled_scores.csv',
                               key=f'scaled_{up.name}', use_container_width=True)

    # downloads
    xlsx_path = os.path.join(tempfile.gettempdir(), f'irt_{mock_id}_abc.xlsx')
    write_results_xlsx(res, xlsx_path)
    all_df = pd.concat([pd.DataFrame(res[s]['items']) for s in subjects], ignore_index=True)
    d1, d2 = st.columns(2)
    with open(xlsx_path, 'rb') as fh:
        d1.download_button('⬇️ Download Excel (a,b,c + detail)', fh.read(),
                           file_name=f'mock_{mock_id}_abc.xlsx', key=f'xlsx_{up.name}')
    d2.download_button('⬇️ Download CSV (all items)', all_df.to_csv(index=False),
                       file_name=f'mock_{mock_id}_items.csv', key=f'csv_{up.name}')
    st.divider()
