"""
Digital SAT IRT Scoring Studio  —  Streamlit app.

Upload a mock-test response CSV (student names + item responses) and get:
  * 3PL item parameters (a, b, c) with QA flags, calibrated from the responses
  * EAP ability (theta) + CSEM per student
  * scaled SAT section scores (200-800) and total (400-1600)

Scoring is norm-referenced: score = mean + SD * theta, where theta is the
student's IRT ability. Enter the mean & SD per subject; the cohort then centres
on that mean. (Absolute-scale anchoring to an external reference is a separate,
one-time step — see the README.)

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from irt_engine import (MMLECalibrator, Priors, eap_scores, prob_3pl,
                        load_form_bytes)

SUBJECT_LABEL = {"reading_and_writing": "Reading & Writing", "math": "Math"}
DEFAULT_SCALE = {"reading_and_writing": (500, 100), "math": (500, 100)}

st.set_page_config(page_title="SAT IRT Scoring Studio", page_icon="📊",
                   layout="wide")

# ------------------------------- styling ---------------------------------- #
st.markdown("""
<style>
:root { --brand:#2f5bea; --ink:#1f2733; --muted:#5b6673; }
.block-container {padding-top:1.4rem; max-width:1320px;}
h1,h2,h3 {color:var(--ink);}
/* hero banner */
.hero {background:linear-gradient(100deg,#2f5bea,#5b7bf0);
       color:#fff; padding:20px 26px; border-radius:16px; margin-bottom:18px;}
.hero h1 {color:#fff; margin:0; font-size:1.55rem; font-weight:700;}
.hero p  {color:#e7edff; margin:.25rem 0 0; font-size:.95rem;}
/* sidebar: light + readable */
[data-testid="stSidebar"] {background:#f4f6fb; border-right:1px solid #e4e9f2;}
[data-testid="stSidebar"] * {color:var(--ink) !important;}
/* make every input clearly visible: white field, dark text, crisp border */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] .stNumberInput input {
    background:#ffffff !important; color:var(--ink) !important;
    border:1px solid #c7d0e0 !important; border-radius:8px !important;
    font-weight:600 !important; font-size:1rem !important;}
[data-testid="stSidebar"] label {font-weight:600 !important; color:var(--ink) !important;}
.stNumberInput button {background:#eef2fb !important;}
div[data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e6ebf2;
    border-radius:12px; padding:12px 16px;}
.subjcard {background:#fff; border:1px solid #e4e9f2; border-radius:12px;
    padding:12px 14px 4px; margin-bottom:10px;}
.subjcard h4 {margin:.1rem 0 .4rem; color:var(--brand); font-size:.95rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------- calibration (cached) ------------------------- #
@st.cache_data(show_spinner=False)
def calibrate(file_bytes: bytes):
    """Calibrate every subject in the uploaded file. Cached on file content so
    changing mean/SD re-scales instantly WITHOUT recalibrating."""
    forms = load_form_bytes(file_bytes)
    result = {}
    for subject, fd in forms.items():
        res = MMLECalibrator(D=1.702, max_iter=300, tol=5e-4,
                             priors=Priors()).fit(fd.responses,
                                                  item_ids=fd.item_ids)
        theta, se = eap_scores(fd.responses, res.a, res.b, res.c, D=res.D)
        var = float(np.var(theta)); merr = float(np.mean(se ** 2))
        result[subject] = dict(
            item_ids=list(map(str, res.item_ids)), sections=list(fd.item_sections),
            a=res.a, b=res.b, c=res.c, N=res.n_administered, pval=res.p_correct,
            flags=res.flags, D=res.D, student_ids=fd.student_ids,
            names=fd.names, emails=fd.emails, theta=theta, se=se,
            reliability=var / (var + merr) if (var + merr) else np.nan)
    return result


def to_scale(theta, mean, sd, lo=200, hi=800, step=10):
    raw = mean + sd * np.asarray(theta)
    return np.clip(np.round(raw / step) * step, lo, hi).astype(int)


# ------------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.markdown("### 📊 SAT IRT Studio")
    st.caption("3PL calibration + IRT scoring, straight from responses.")
    up = st.file_uploader("Upload response CSV", type=["csv"],
                          help="Student names + item responses (1 / 0 / -).")
    st.markdown("---")
    st.markdown("#### Scaling per subject")
    st.caption("score = **mean + SD × ability**. The cohort centres on *mean*.")

    st.markdown('<div class="subjcard"><h4>Reading & Writing</h4>',
                unsafe_allow_html=True)
    cr = st.columns(2)
    rw_mean = cr[0].number_input("Mean", 200, 800, DEFAULT_SCALE["reading_and_writing"][0],
                                 10, key="rw_mean")
    rw_sd = cr[1].number_input("SD", 10, 300, DEFAULT_SCALE["reading_and_writing"][1],
                               5, key="rw_sd")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="subjcard"><h4>Math</h4>', unsafe_allow_html=True)
    cm = st.columns(2)
    ma_mean = cm[0].number_input("Mean", 200, 800, DEFAULT_SCALE["math"][0],
                                 10, key="ma_mean")
    ma_sd = cm[1].number_input("SD", 10, 300, DEFAULT_SCALE["math"][1], 5,
                               key="ma_sd")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Model: 3PL · MMLE-EM · D = 1.702 · MAP priors")

MEANSD = {"reading_and_writing": (rw_mean, rw_sd), "math": (ma_mean, ma_sd)}

# ------------------------------- hero ------------------------------------- #
st.markdown("""
<div class="hero">
  <h1>Digital SAT — IRT Scoring Studio</h1>
  <p>Upload student responses → calibrated item parameters (a, b, c) and scaled SAT scores.</p>
</div>""", unsafe_allow_html=True)

if up is None:
    st.info("⬅️ **Upload a response CSV to begin.** You'll get item parameters "
            "(a, b, c) and scaled SAT scores. Set each subject's mean & SD in "
            "the sidebar — scores re-scale instantly, no recalculation.")
    st.stop()

with st.spinner("Calibrating 3PL item parameters (MMLE-EM)…"):
    cal = calibrate(up.getvalue())
subjects = list(cal.keys())

# ------------------------- assemble student scores ------------------------ #
frames = []
for subject in subjects:
    d = cal[subject]
    mean, sd = MEANSD[subject]
    frames.append(pd.DataFrame({
        "Student ID": d["student_ids"], "Name": d["names"], "Email": d["emails"],
        f"{SUBJECT_LABEL[subject]} score": to_scale(d["theta"], mean, sd),
        f"θ_{subject}": np.round(d["theta"], 3),
        f"CSEM_{subject}": np.round(d["se"], 3)}))
scores = frames[0]
for f in frames[1:]:
    scores = scores.merge(f, on=["Student ID", "Name", "Email"], how="outer")
score_cols = [c for c in scores.columns if c.endswith("score")]
if len(score_cols) == 2:
    scores["Total (400–1600)"] = scores[score_cols].fillna(0).sum(axis=1).astype(int)
    total_col = "Total (400–1600)"
else:
    total_col = score_cols[0]

# ------------------------------- KPI row ---------------------------------- #
k = st.columns(4)
k[0].metric("Students", f"{scores['Name'].nunique():,}")
k[1].metric("Items calibrated", sum(len(cal[s]['a']) for s in subjects))
k[2].metric(f"Mean {total_col.split(' ')[0].lower()}", f"{scores[total_col].mean():.0f}")
flagged = sum(sum(f != 'ok' for f in cal[s]['flags']) for s in subjects)
k[3].metric("Items flagged (QA)", flagged)

tab_scores, tab_items, tab_diag = st.tabs(
    ["🎓 Student Scores", "📐 Item Parameters", "🩺 Diagnostics"])

with tab_scores:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Scored students")
        st.dataframe(scores.sort_values(total_col, ascending=False),
                     use_container_width=True, height=460, hide_index=True)
        st.download_button("⬇️ Download scores (CSV)",
                           scores.to_csv(index=False).encode(),
                           "student_scores.csv", "text/csv")
    with c2:
        st.subheader(f"{total_col} distribution")
        fig = px.histogram(scores, x=total_col, nbins=30,
                           color_discrete_sequence=["#2f5bea"])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"mean {scores[total_col].mean():.0f} · SD "
                   f"{scores[total_col].std():.0f} · min {scores[total_col].min()} · "
                   f"max {scores[total_col].max()}")

with tab_items:
    subject = st.radio("Subject", subjects,
                       format_func=lambda s: SUBJECT_LABEL[s], horizontal=True)
    d = cal[subject]
    idf = pd.DataFrame({
        "Item ID": d["item_ids"], "Section": d["sections"],
        "a": np.round(d["a"], 3), "b": np.round(d["b"], 3),
        "c": np.round(d["c"], 3), "N": d["N"],
        "p-value": np.round(d["pval"], 3), "QA flag": d["flags"]})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(idf, use_container_width=True, height=460, hide_index=True)
        st.download_button("⬇️ Download item parameters (CSV)",
                           idf.to_csv(index=False).encode(),
                           f"item_params_{subject}.csv", "text/csv")
    with c2:
        st.markdown("**Item characteristic curves**")
        theta = np.linspace(-4, 4, 100)
        fig = go.Figure()
        for i in range(len(d["a"])):
            P = prob_3pl(theta, d["a"][i], d["b"][i], d["c"][i], d["D"])
            fig.add_trace(go.Scatter(x=theta, y=P, mode="lines",
                                     line=dict(width=1, color="#2f5bea"),
                                     opacity=0.3, hoverinfo="skip", showlegend=False))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="ability θ", yaxis_title="P(correct)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each curve = one item. a = slope, b = location, c = lower asymptote.")

with tab_diag:
    st.subheader("Reliability & data quality")
    for subject in subjects:
        d = cal[subject]
        cols = st.columns(4)
        cols[0].metric(f"{SUBJECT_LABEL[subject]} — reliability",
                       f"{d['reliability']:.3f}")
        cols[1].metric("Median CSEM", f"{np.median(d['se']):.3f}")
        cols[2].metric("Low-N items (N<50)", int(np.sum(d["N"] < 50)))
        cols[3].metric("Too-easy items (p>.95)",
                       int(np.sum(np.array(d["pval"]) > 0.95)))
    st.divider()
    st.markdown("""
**Reading these numbers**
- **Reliability 0.75–0.90** is normal for a single adaptive module length.
- **Low-N items** come from barely-routed Module-2 forms (few students saw them);
  their a/b/c are prior-pulled and affect only those few students.
- **Too-easy items** have poorly-identified discrimination `a` by nature.
- Scores are **norm-referenced** to the mean/SD you set — the cohort averages the
  mean you entered. Matching an external absolute scale is a one-time anchoring
  step, separate from calibration.
""")
