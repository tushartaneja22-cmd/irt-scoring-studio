"""
IRT Scoring Studio - Streamlit front-end for irt_pipeline.py

Run:
    pip install streamlit pandas openpyxl scipy numpy
    streamlit run app.py

Upload a response file (CSV or Excel), set each section's Mean and Standard
Deviation, and download calibrated item parameters + scaled student scores.
Heavy IRT calibration + WLE scoring runs ONCE (cached); changing Mean/SD
re-scales instantly.
"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import irt_pipeline as engine

# ----------------------------------------------------------------------
# Page setup + styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="IRT Scoring Studio", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 2.2rem; max-width: 1200px;}
      html, body, [class*="css"] {font-family: 'Inter','Segoe UI',system-ui,sans-serif;}
      .hero {
        background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#9333ea 100%);
        border-radius: 18px; padding: 26px 32px; color: #fff; margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(79,70,229,.28);
      }
      .hero h1 {font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing:-.02em;}
      .hero p  {font-size: .98rem; opacity: .92; margin: 0;}
      .card {
        background: var(--background-color,#fff); border: 1px solid rgba(120,120,140,.18);
        border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 10px rgba(0,0,0,.04);
      }
      .stat {
        border-radius: 14px; padding: 16px 18px; color:#fff; text-align:center;
        box-shadow: 0 6px 18px rgba(0,0,0,.10);
      }
      .stat .v {font-size: 1.7rem; font-weight: 700; line-height:1.1;}
      .stat .l {font-size: .78rem; text-transform: uppercase; letter-spacing:.06em; opacity:.9;}
      .s1{background:linear-gradient(135deg,#6366f1,#4f46e5);}
      .s2{background:linear-gradient(135deg,#0ea5e9,#0284c7);}
      .s3{background:linear-gradient(135deg,#10b981,#059669);}
      .s4{background:linear-gradient(135deg,#f59e0b,#d97706);}
      .sectlabel{font-weight:600;font-size:.95rem;margin:.2rem 0 .1rem;}
      .stDownloadButton button {border-radius:10px;font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎯 IRT Scoring Studio</h1>
      <p>Upload responses → calibrate 3PL item parameters → score students with WLE.
         Set each section's scale (mean &amp; SD) and export to CSV or Excel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Heavy engine call (cached): calibrate + WLE theta (no scaling yet)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_engine(file_bytes: bytes, file_name: str, sheet: str, free_math_c: bool = False):
    suffix = os.path.splitext(file_name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        header, body = engine.read_table(path, sheet)
        item_cols = engine.detect_item_columns(header, body)
        if not item_cols:
            raise ValueError("No question columns detected in this file.")
        item_ids = [str(header[c]) for c in item_cols]
        item_subject = np.array([engine.subject_of(header[c], engine.DEFAULT_SUBJECT_KEYWORDS)
                                 for c in item_cols])
        n_items, n_students = len(item_cols), len(body)
        resp = np.zeros((n_items, n_students)); mask = np.zeros((n_items, n_students))
        for j, c in enumerate(item_cols):
            for i, row in enumerate(body):
                v = engine._resp_val(row[c] if c < len(row) else None)
                if v is not None:
                    resp[j, i] = v; mask[j, i] = 1
        ids = [row[0] if len(row) else None for row in body]
        names = [row[1] if len(row) > 1 else None for row in body]

        subjects = list(dict.fromkeys(item_subject.tolist()))
        free_mask = (item_subject == "Math") if free_math_c else None
        a, b, c, se, n_adm, it, conv = engine.calibrate(resp, mask, verbose=False, free_c_mask=free_mask)
        scored = engine.score_students(resp, mask, item_subject, a, b, c, subjects)
    finally:
        os.unlink(path)

    return {
        "item_ids": item_ids, "item_subject": item_subject.tolist(),
        "a": a, "b": b, "c": c, "se": se, "n_adm": n_adm,
        "ids": ids, "names": names,
        "subjects": subjects,
        "theta": {s: scored[s][0] for s in subjects},
        "nit": {s: scored[s][1] for s in subjects},
        "n_items": n_items, "n_students": n_students, "converged": conv, "iters": it,
        "free_math_c": free_math_c,
    }


def build_frames(res, scale):
    # Item parameters
    items = []
    for j, qid in enumerate(res["item_ids"]):
        n = int(res["n_adm"][j])
        if n == 0:
            items.append([qid, res["item_subject"][j], None, None, None, None, 0, "None"])
        else:
            se = res["se"][j]
            items.append([qid, res["item_subject"][j], round(float(res["a"][j]), 3),
                          round(float(res["b"][j]), 3), round(float(res["c"][j]), 3),
                          round(float(se), 3) if np.isfinite(se) else "n/a", n, engine._conf(n)])
    df_items = pd.DataFrame(items, columns=["Question ID", "Subject", "a (Discrimination)",
                                            "b (Difficulty)", "c (Guessing)", "SE (b)",
                                            "n students", "Confidence"])
    # Student scores
    subjects = res["subjects"]
    rows = []
    for i in range(res["n_students"]):
        row = {"Id": res["ids"][i], "Name": res["names"][i]}
        total = 0
        for s in subjects:
            theta = res["theta"][s][i]; nit = int(res["nit"][s][i])
            row[f"{s} items"] = nit
            if np.isnan(theta):
                row[f"{s} theta"] = None; row[f"{s} score"] = None
            else:
                xb, sg = scale[s]
                sc = int(engine.to_scaled(np.array([theta]), xb, sg)[0])
                row[f"{s} theta"] = round(float(theta), 3); row[f"{s} score"] = sc
                total += sc
        row["Total score"] = total
        rows.append(row)
    df_scores = pd.DataFrame(rows)
    return df_items, df_scores


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------
up = st.file_uploader("Upload responses (CSV or Excel — one row per student, 1/0/– per question)",
                      type=["csv", "tsv", "xlsx", "xlsm"])

if up is None:
    st.info("👋 Upload a response file to begin. Columns: an ID, an optional Name, "
            "then one column per question with **1** (correct), **0** (wrong), or **–** (not administered).")
    st.stop()

sheet = "Student Responses"
if up.name.lower().endswith((".xlsx", ".xlsm")):
    sheet = st.text_input("Excel sheet name", value="Student Responses")

free_math_c = st.toggle(
    "🔬 Free math guessing (Option A) — let SPR / grid-in items self-identify (c → 0)",
    value=False,
    help="Default OFF keeps c ≈ 0.25 for every multiple-choice item (current behavior). When ON, math "
         "items get a relaxed guessing prior so student-produced-response (grid-in) questions fall to "
         "near-zero guessing automatically, while true MCQs stay near 0.25. Toggling re-runs calibration.",
)

with st.spinner("Calibrating item parameters and estimating abilities… (runs once per setting, then cached)"):
    try:
        res = run_engine(up.getvalue(), up.name, sheet, free_math_c)
    except Exception as e:
        st.error(f"Could not process file: {e}")
        st.stop()

subjects = res["subjects"]
st.success(f"Calibrated **{res['n_items']} questions** for **{res['n_students']} students** "
           f"({'converged' if res['converged'] else 'stopped at cap'} in {res['iters']} iterations). "
           f"Sections detected: {', '.join(subjects)}.")

# ----------------------------------------------------------------------
# Scale controls (Mean / SD per section) — you set these
# ----------------------------------------------------------------------
st.markdown("#### ⚙️ Score scale — set the Mean and Standard Deviation per section")
st.caption("Reported score = **Mean + SD × θ**, rounded to 10 and clamped to 200–800. "
           "Changing these re-scales instantly (no re-calibration).")

scale = {}
cols = st.columns(max(1, len(subjects)))
for k, s in enumerate(subjects):
    d_xb, d_sg = engine.DEFAULT_SCALE.get(s, engine.DEFAULT_SCALE["Other"])
    with cols[k]:
        st.markdown(f"<div class='sectlabel'>{s}</div>", unsafe_allow_html=True)
        xb = st.number_input(f"{s} — Mean", value=float(d_xb), step=5.0, key=f"xb_{s}")
        sg = st.number_input(f"{s} — Std. Dev.", value=float(d_sg), min_value=1.0, step=5.0, key=f"sg_{s}")
        scale[s] = (xb, sg)

df_items, df_scores = build_frames(res, scale)

# ----------------------------------------------------------------------
# Summary stats
# ----------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
palette = ["s1", "s2", "s3", "s4"]
metrics = [("Students", f"{res['n_students']}"), ("Questions", f"{res['n_items']}")]
if "Total score" in df_scores and df_scores["Total score"].notna().any():
    metrics.append(("Mean total", f"{df_scores['Total score'].mean():.0f}"))
    metrics.append(("Total range", f"{int(df_scores['Total score'].min())}–{int(df_scores['Total score'].max())}"))
mc = st.columns(len(metrics))
for k, (label, val) in enumerate(metrics):
    with mc[k]:
        st.markdown(f"<div class='stat {palette[k % 4]}'><div class='v'>{val}</div>"
                    f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Results tabs
# ----------------------------------------------------------------------
t_scores, t_items, t_dist = st.tabs(["📋 Student Scores", "🎯 Item Parameters", "📈 Distribution"])

with t_scores:
    st.dataframe(df_scores, use_container_width=True, height=430)
    st.download_button("⬇️ Download scores (CSV)", df_scores.to_csv(index=False).encode("utf-8"),
                       "student_scores.csv", "text/csv")

with t_items:
    st.dataframe(df_items, use_container_width=True, height=430)
    st.download_button("⬇️ Download item parameters (CSV)", df_items.to_csv(index=False).encode("utf-8"),
                       "item_parameters.csv", "text/csv")
    if res.get("free_math_c"):
        st.markdown("---")
        st.caption("🔬 Option A is ON — math guessing (c) was estimated freely. "
                   "Items with very low c are likely SPR / grid-in; genuine MCQs stay near 0.25.")
        math_df = df_items[df_items["Subject"] == "Math"].copy()
        cnum = pd.to_numeric(math_df["c (Guessing)"], errors="coerce")
        math_df["Likely SPR"] = np.where(cnum < 0.10, "✅ grid-in?", "")
        n_spr = int((cnum < 0.10).sum())
        st.markdown(f"**Math items by estimated guessing (c), lowest first — {n_spr} flagged as likely grid-in**")
        st.dataframe(math_df.sort_values("c (Guessing)", na_position="last"),
                     use_container_width=True, height=330)

with t_dist:
    score_cols = [f"{s} score" for s in subjects if f"{s} score" in df_scores]
    plot_cols = score_cols + (["Total score"] if "Total score" in df_scores else [])
    pick = st.selectbox("Score to plot", plot_cols, index=len(plot_cols) - 1 if plot_cols else 0)
    vals = pd.to_numeric(df_scores[pick], errors="coerce").dropna()
    if len(vals):
        hist = np.histogram(vals, bins=np.arange(200, 1610, 40))
        chart_df = pd.DataFrame({"count": hist[0]}, index=[int(x) for x in hist[1][:-1]])
        st.bar_chart(chart_df, height=340, color="#7c3aed")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{vals.mean():.0f}")
        c2.metric("Median", f"{vals.median():.0f}")
        c3.metric("Std. Dev.", f"{vals.std():.0f}")

# ----------------------------------------------------------------------
# Combined Excel (two tabs)
# ----------------------------------------------------------------------
st.markdown("---")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    df_items.to_excel(writer, sheet_name="Item Parameters", index=False)
    df_scores.to_excel(writer, sheet_name="Student Scores", index=False)
st.download_button("⬇️ Download full workbook (Excel, both tabs)", buf.getvalue(),
                   "irt_results.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
