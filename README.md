# 🎯 IRT Scoring Studio

A Streamlit app that turns raw test responses into calibrated item parameters and scaled student scores using Item Response Theory (3PL).

Upload a response file → the app **calibrates** each question's difficulty/discrimination/guessing (3PL, Marginal Maximum Likelihood) and **scores** every student's ability with **WLE** (Warm's Weighted Likelihood) → download the results as CSV or a two‑tab Excel workbook.

## Features

- **Calibration (Stage 1):** 3PL item parameters (a, b, c) via Bock–Aitkin EM, with `D = 1.702` (normal‑ogive metric) and mild priors on a/b/c for stability on sparse or extreme items.
- **Scoring (Stage 2):** per‑section ability θ via WLE, then `score = Mean + SD × θ`, rounded to the nearest 10 and clamped to 200–800.
- **You control the scale:** set each section's **Mean** and **Standard Deviation** in the UI. Calibration is cached, so changing the scale re‑scores instantly.
- **Reusable:** auto‑detects question columns and sections, so it works on any test — not just one template.
- **Exports:** Student Scores CSV, Item Parameters CSV, or a combined two‑tab Excel workbook.

## Input format

One row per student. Columns:

- an **ID** column, an optional **Name** column, then
- one column per **question** — cell = `1` (correct), `0` (wrong), or `-`/blank (not administered, e.g. an adaptive form the student didn't take).

Sections are inferred from question‑column names (math vs reading/writing/verbal). No data files are stored in this repo — everything is uploaded at runtime.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Command‑line (no UI)

`irt_pipeline.py` also runs standalone:

```bash
python irt_pipeline.py responses.csv --scale "R&W:670:60,Math:710:70"
```

It writes `<input>_results.xlsx` with the two tabs.

## Deploy

Deploys as‑is to [Streamlit Community Cloud](https://streamlit.io/cloud): point it at this repo and `app.py`. No secrets or data required.

## Method notes

- The latent ability scale is fixed internally to mean 0 / SD 1 (standard IRT identification). The section Mean/SD only convert θ to a reported score.
- WLE is chosen over EAP/MLE for individual score reporting: it is bias‑corrected and defined even for perfect (all‑right/all‑wrong) papers.
