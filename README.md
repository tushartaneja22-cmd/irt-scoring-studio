# IRT Scoring Studio

Digital SAT **3PL item calibration** — produce `a` (discrimination), `b` (difficulty),
and `c` (guessing) for every item on a mock, from the raw response export, fast and
reproducibly, and matched to the reference (gold-standard) metric.

---

## What it does

Given a mock CSV in the standard export layout (metadata columns, then item columns
like `S1_24667_(STANDARD)(reading_and_writing)` with `1`/`0`/`-` cells), the studio:

1. Parses each subject (Reading & Writing, Math) into a student × item response matrix.
   Adaptive Module-2 routing (hard vs. easy variant) shows up as `-` and is treated as
   **missing-by-design**, which is ignorable under marginal-maximum-likelihood.
2. **Calibrates a 3PL model** by MML-EM (Bock–Aitkin) on a fixed `N(0,1)` trait,
   concurrently over all items of a subject, with MAP priors that stabilise the small,
   unevenly-administered adaptive samples:
   - `log(a) ~ N(log 0.80, 0.25)` (slope)
   - `logit(c) ~ N(logit 0.25, 0.25)` (guessing centred at 1/4 for 4-option MC)
   - weak ridge on the intercept.
3. **Links to the reference metric.** The raw MML output ranks items well but lives on a
   different scale than the reference workbook. A small per-subject linear model
   (frozen in `engine/link_model.json`, trained on all six reference mocks) maps
   `(MML params + classical item statistics) → reference a, b, c`.
4. **Flags data-quality issues** — items seen by too few students (thin adaptive modules)
   or with no response variance are marked `low_n` / `degenerate`.

## Two engines

The reference workbook was produced by **xCalibre** (Assessment Systems Corp.). The studio
offers two calibration engines:

- **`link`** (default) — MML-EM 3PL on a fixed `N(0,1)` trait, then a learned metric link to
  the reference scale. **Lowest held-out RMSE**, fast (~4 s/mock).
- **`xcalibre`** — reproduces xCalibre's estimator conventions directly: normal-ogive
  **D = 1.702** discrimination metric, **floating (empirical-Bayes) priors** on `a` and `b`
  re-estimated each EM cycle, and a fixed guessing prior at `1/#options`, followed by a light
  2-constant metric anchor. More interpretable and mechanistically faithful; recovers a better
  *within-mock* discrimination ranking (r ≈ 0.6–0.75 vs ≈ 0.47), but its `a`-scale is
  mock-dependent so cross-mock RMSE is comparable, not better. Slower (~35 s/mock).

Select with `--mode` on the CLI or the sidebar toggle in the app.

## Accuracy (honest leave-one-mock-out cross-validation)

Measured on well-administered items (≥50 students), holding each mock out and linking on
the others — i.e. the accuracy expected on a **brand-new** mock:

| Parameter | Reading & Writing | Math |
|-----------|------------------|------|
| `b` difficulty | RMSE ≈ 0.44, r ≈ 0.96 | RMSE ≈ 0.44, r ≈ 0.94 |
| `c` guessing   | RMSE ≈ 0.015 | RMSE ≈ 0.022 |
| `a` discrimination | RMSE ≈ 0.13 | RMSE ≈ 0.20 |

The `b` link uses a **continuity-corrected, guessing-adjusted difficulty feature** (plus a
cubic term and a ±4 clamp) so that extreme easy/hard items — where the raw proportion
saturates — still reach the reference's extreme difficulties. This cut the Math tail
(p ≥ 0.9 or ≤ 0.15) b-RMSE by ~17%.

`c` is essentially exact. `b` tracks the gold standard tightly. `a` is intrinsically
hard to recover here because the reference `a` values are heavily shrunk (sd ≈ 0.12) — the
calibrator matches their central tendency, and the residual signal is small for everyone.
Lightly-taken adaptive modules (e.g. mock 117 Math's hard module, 12–17 takers) are
flagged and unavoidably noisier.

## Artificial correction toggle

Each mock's raw output carries a residual deviation from the reference that is two
things stacked: a **per-mock offset/scale wobble** (mostly in `a`, and some mocks'
`b`) and **irreducible item-level scatter** (`b`'s ~0.4 RMSE is just the r ≈ 0.96
residual, `b` sd ≈ 1.3 → 1.3·√(1−0.96²) ≈ 0.37 — not a bias). The offsets *flip
sign across mocks*, so a learned global correction fitted on other mocks makes a new
mock **worse** out-of-sample (every leave-one-out affine fit raised RMSE). The only
lever that reduces error on a given mock uses that mock's **own** reference values.

That is what the **artificial correction** does — it is gold-anchored, so it needs the
reference workbook loaded and applies only to **referenced** mocks. Use it to
reproduce / audit reference values, not to score a brand-new un-referenced mock. Per
subject × parameter, modes escalate:

| Mode | Transform | Keeps rank? | Effect |
|------|-----------|-------------|--------|
| `off` | identity | — | raw studio output |
| `bias` | `+ (mean_gold − mean_est)` | yes | matches the reference **mean** |
| `moment` | z-score to gold mean **and** sd | yes | matches first two moments |
| `regress` | oracle LSQ `gold ~ k·est + m` | yes | **minimum RMSE** for the ranking |
| `exact` | snap to gold | n/a | RMSE = 0 (reproduce the sheet) |

A **strength** slider (0–1) blends raw → corrected. `regress` is the safe choice
(never worse than `off` on RMSE, and it shrinks its slope on low-correlation params
like `a`/`c` instead of amplifying noise the way `moment` can); `exact` is the full
snap. Example on mock 119 Math `b`: RMSE 0.374 (off) → 0.302 (bias) → 0.272
(regress) → 0 (exact). Corrected workbooks keep `*_uncorrected`, `*_gold`, and a
`corrected` label column so every adjustment is auditable.

App: **sidebar → Artificial correction** (mode + strength). CLI: `--correct
{off,bias,moment,regress,exact} --strength 0..1` (with `--gold`).

## Usage

### Interactive app
```bash
streamlit run app/app.py
```
Upload one or more mock CSVs (optionally the gold `A, B, C Values.xlsx` to see live
validation). View per-item `a/b/c` with QC flags, difficulty/discrimination charts, and
download Excel or CSV.

### Batch CLI
```bash
python engine/cli.py "Digital_SAT_Mock_Test_-115_1754312309.csv" \
    --gold "A, B, C Values.xlsx" --out output
python engine/cli.py *.csv --out output                 # many mocks at once
python engine/cli.py *.csv --out output --mode xcalibre # xCalibre-faithful engine
python engine/cli.py *.csv --gold "A, B, C Values.xlsx" \
    --out output --correct regress                       # artificial gold-anchored correction
```

### Re-validate everything / re-fit the link
```bash
python validate_all.py        # calibrate all 6 mocks, print accuracy, write output/
python engine/cv.py           # leave-one-out cross-validation
python engine/fit_link.py     # refit + freeze engine/link_model.json (e.g. after adding mocks)
```

## Layout

```
engine/
  loader.py      CSV -> response matrices (per subject)
  calibrate.py   3PL MML-EM calibrator (vectorised E-step, analytic-gradient M-step)
  xcalibre.py    xCalibre-faithful engine (normal-ogive D=1.702 + floating priors)
  features.py    classical item statistics used by the link
  link.py        fit / apply / save-load the metric-link model
  link_model.json      frozen link coefficients (trained on mocks 115-120)
  xcalibre_anchor.json frozen 2-constant metric anchor for the xcalibre engine
  pipeline.py    calibrate -> link/anchor -> QC, per mock (mode= link | xcalibre)
  report.py      Excel/CSV writers + validation metrics
  cli.py         batch command line
  gold.py        read reference a/b/c workbook
  tune.py, cv.py, fit_link.py   tuning / cross-validation / freezing utilities
app/app.py       Streamlit UI
validate_all.py  full run + accuracy report
output/          generated workbooks
```

## Extending to future mocks

The pipeline needs **no gold standard** to score a new mock — just run it. The frozen link
generalises (that is what the cross-validation measures). If you later obtain reference
values for additional mocks and want to sharpen the link, drop them in, add their ids to
`fit_link.py`, and re-run it to refresh `link_model.json`.

## Method notes / assumptions

- Unidimensional 3PL per subject; concurrent calibration across Module 1 + both Module 2
  variants on one trait.
- The reference workbook lists `a,b,c` in item-column order with no ids; positional
  alignment holds only where the reference kept every item. Where it dropped thin items,
  the studio still reports all items (flagged) but skips positional validation.
- `c` is priored at 0.25 (four-option multiple choice); adjust in the app sidebar if a form
  uses a different option count.
