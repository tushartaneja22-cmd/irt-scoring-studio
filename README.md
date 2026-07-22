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

The reference was produced by **xCalibre** (Assessment Systems Corp.). The studio
offers two calibration engines:

- **`link`** (default) — MML-EM 3PL on a fixed `N(0,1)` trait, then a learned metric link to
  the reference scale. **Lowest held-out RMSE.** For the **Math** subject it also runs the
  xcalibre engine and feeds its slope (`xc_a`) into the `a` link, which recovers the reference
  Math discrimination markedly better (see Accuracy); RW stays MML-only. ~15–40 s/mock.
- **`xcalibre`** — reproduces xCalibre's estimator conventions directly: normal-ogive
  **D = 1.702** discrimination metric, **floating (empirical-Bayes) priors** on `a` and `b`
  re-estimated each EM cycle, and a fixed guessing prior at `1/#options`, followed by a light
  2-constant metric anchor. More interpretable and mechanistically faithful; recovers a better
  *within-mock* discrimination ranking (r ≈ 0.6–0.75 vs ≈ 0.47), but its `a`-scale is
  mock-dependent so cross-mock RMSE is comparable, not better. Slower (~35 s/mock).

Select with `--mode` on the CLI or the sidebar toggle in the app.

## Reference format

The ground-truth `a/b/c` are the per-mock **ID-keyed JSON exports** (`<mock-id>.txt`, e.g.
`115.txt`), whose `questions[]` array carries each item's `question` id and `irt_a/irt_b/irt_c`.
The studio aligns estimates to the reference **by question id**, so every item on every mock
is used — no positional / dropped-item misalignment. (This replaced an earlier positional
Excel workbook.) `engine/refjson.py` loads and aligns them.

## Accuracy (honest leave-one-mock-out cross-validation)

Holding each mock out and linking on the others — i.e. the accuracy expected on a
**brand-new** mock. Measured against the ID-keyed JSON reference:

| Parameter | Reading & Writing | Math |
|-----------|------------------|------|
| `b` difficulty | r ≈ 0.93, RMSE ≈ 0.56 | r ≈ 0.87, RMSE ≈ 0.64 |
| `c` guessing   | r ≈ 0.17, RMSE ≈ 0.033 | RMSE ≈ 0.051 |
| `a` discrimination | **r ≈ 0.45, RMSE ≈ 0.157** | **r ≈ 0.39, RMSE ≈ 0.211** |

(Well-administered items, ≥50 students. `python engine/cv.py` prints the full table.)

### The `a` link (per subject)

Discrimination is the hard parameter — the reference `a` is heavily shrunk (sd ≈ 0.18 RW /
0.22 Math), so much of it is prior, not item-level signal. The two subjects are recovered
differently, so the `a` link now uses **per-subject features** (`engine/link.py`):

- **Reading & Writing** — the reference couples discrimination to difficulty, so RW `a` is
  best recovered from the MML slope plus classical discrimination proxies (biserial,
  point-biserial) **and guessing-adjusted difficulty** (`b`, `zpc`). All cheap, MML-only.
  Held-out `a` r **0.29 → 0.42**.
- **Math** — the reference discrimination ranking is recovered markedly better by the
  **xCalibre-faithful slope** (normal-ogive + floating priors), so Math `a` adds the `xc_a`
  feature. The pipeline runs the xcalibre engine **for the Math subject only** at inference.
  Held-out `a` r **0.28 → 0.40**.

Both gains are leave-one-mock-out validated (honest, no gold-peeking) and stable across all
six held-out mocks. `b` and `c` are unchanged.

`c` is essentially at its floor (near-constant ≈ 0.25). `b` tracks the reference tightly.
`a` matches ranking and central tendency; because the signal is weak the linked `a` is more
shrunk than the reference (it minimises squared error rather than matching the spread), so
the extreme-discrimination items are pulled toward the mean — an honest limit, not a bug.
Lightly-taken adaptive modules (e.g. mock 120) are flagged and unavoidably noisier.

**Runtime.** Because Math now also runs the xcalibre engine in `link` mode, a full mock takes
~15–40 s (was ~4 s). RW stays fast (MML-only).

## Scaled scores

Beyond item parameters, the app scores **students**. From the calibrated 3PL pool it
estimates each student's latent ability **θ by EAP** (expected a-posteriori) on the same
`N(0,1)` trait, using only that student's administered items (robust to adaptive routing,
and finite even for all-correct / all-incorrect students). θ is then mapped to a Digital-SAT
section scale:

```
section = round( mean + sd·θ )  clamped to [200, 800], rounded to 10   (default mean 500, sd 100)
total   = RW + Math             (400–1600)
```

The **Scaled scores** tab shows every student's Id, name, section scores, total, and items
answered, with a distribution and CSV download; the section mean/sd are adjustable in the
sidebar. This is a **transparent norm-referenced** conversion (θ standardised on the cohort),
**not** the College Board's official, proprietary, form-specific raw-to-scaled table — treat
it as a well-calibrated relative score, not an official SAT score.

## Why there is no "correction toggle"

It is tempting to add a post-hoc knob that nudges the output onto the reference values.
We tested this rigorously and **removed it from the app**, because the only version that
improves accuracy is dishonest for real use:

- The deviation from the reference is two things stacked: a **per-mock offset** (mostly
  in `a`) and **irreducible item-level scatter** (`b`'s ~0.4 RMSE is just the r ≈ 0.96
  residual — `b` sd ≈ 1.3, so 1.3·√(1−0.96²) ≈ 0.37 — not a bias a formula can remove).
- The per-mock offsets **flip sign across mocks**, so any correction learned from other
  mocks and applied gold-free makes a **new** mock *worse*. Every leave-one-out variant
  (match-mean, match-mean+sd, affine regress) raised held-out RMSE.
- The only thing that lowers a given mock's error uses that mock's **own** reference
  answer key — which is circular (if you have the a/b/c, you don't need to estimate them).

So the honest, generalising correction is simply the calibration itself (below). A
gold-anchored alignment remains available as a **developer/QA tool** in the CLI
(`--correct {bias,moment,regress,exact} --strength 0..1`, requires `--gold`) for
*reproducing/auditing* the six reference mocks — it is deliberately **not** in the app.

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
  features.py    classical item statistics + optional xCalibre features used by the link
  link.py        fit / apply / save-load the metric-link model (per-subject `a` columns)
  link_model.json      frozen link coefficients (trained on mocks 115-120)
  xcalibre_anchor.json frozen 2-constant metric anchor for the xcalibre engine
  pipeline.py    calibrate -> link/anchor -> QC, per mock (mode= link | xcalibre)
  refjson.py     load + ID-align the reference JSON (`<id>.txt`)
  report.py      Excel/CSV writers + validation metrics
  cli.py         batch command line
  fit_link.py    refit + freeze link_model.json (ID-aligned JSON reference)
  cv.py / validate_loo.py   honest leave-one-mock-out validation
  _build_cache.py           cache MML+xCalibre estimates for fast link experiments
  gold.py, tune.py          retired (positional-Excel era; kept for reference)
app/app.py       Streamlit UI
validate_all.py  full run + accuracy report
<id>.txt         per-mock ID-keyed reference (a/b/c by question id)
output/          generated workbooks
```

## Extending to future mocks

The pipeline needs **no reference** to score a new mock — just run it. The frozen link
generalises (that is what the cross-validation measures). If you later obtain reference
values for additional mocks and want to sharpen the link, drop in their `<id>.txt` JSON
exports, add the ids to `engine/refjson.py` (`MOCKS`), delete the stale `engine/_acache.npz`,
and run `python engine/fit_link.py` to refresh `link_model.json`.

## Method notes / assumptions

- Unidimensional 3PL per subject; concurrent calibration across Module 1 + both Module 2
  variants on one trait.
- The reference JSON (`<id>.txt`) carries a `question` id per item, so estimates and
  reference align **by id** — every item on every mock is validated, regardless of ordering.
- `c` is priored at 0.25 (four-option multiple choice); adjust in the app sidebar if a form
  uses a different option count. Note the reference calibrated **all** items (including Math
  student-produced responses) with a guessing parameter ≈ 0.25 — i.e. 3PL throughout, not a
  2PL/GRM for grid-ins — so the studio matches that convention deliberately.
