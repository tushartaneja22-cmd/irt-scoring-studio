# SAT IRT Scoring Engine

A 3-parameter (3PL) Item Response Theory calibration + scoring engine for
Digital SAT adaptive mock-test data. Built to match and exceed commercial
calibrators (Xcalibre, flexMIRT, IRTPRO) on this kind of data.

## Live app / deploy on Streamlit Cloud

The web app is `app.py`. To deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app → From existing repo**, then set:
   - Repository: `tushartaneja22-cmd/irt-scoring-studio`
   - Branch: `main`
   - Main file path: `app.py`
3. Click **Deploy**. Streamlit installs `requirements.txt` and launches the app.

One-click (pre-fills the form — you still click *Deploy*):
`https://share.streamlit.io/deploy?repository=tushartaneja22-cmd/irt-scoring-studio&branch=main&mainModule=app.py`

Run locally instead:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

1. **Parses** the Digital SAT CSV export (adaptive 2-stage: Module 1 anchor +
   routed easy/hard Module 2), treating `-` cells as *missing by design*.
2. **Calibrates** 3PL item parameters `(a, b, c)` per subject
   (Reading & Writing, Math) via **Marginal Maximum Likelihood with EM**
   (Bock–Aitkin) and **MAP priors**.
3. **Scores** every student with **EAP** ability estimates (θ) and a conditional
   standard error (CSEM), then maps θ to the **200–800** section scale and
   **400–1600** total.

## Why it holds up against best-in-class tools

| Capability | Naive run | This engine |
|---|---|---|
| MMLE-EM 3PL core | ✓ | ✓ |
| MAP priors on a, b, c | often off | ✓ (lognormal / normal / Beta) |
| Small-N stability (N≈12–26 routed forms) | diverges | ✓ Beta-c prior + **empirical-Bayes shrinkage** |
| Adaptive missing-by-design | manual setup | ✓ native (NaN skipped, never imputed) |
| θ → official SAT scale | ✗ | ✓ built in |
| Fixed-parameter online scoring of new students | painful | ✓ (`eap_scores` with saved params) |

## Validation 1 — parameter recovery (`validate_recovery.py`)

Blind recovery from known parameters at N≈1500 with adaptive routing injected:

| Parameter | Recovery |
|---|---|
| b (difficulty) | r = 0.97 |
| a (discrimination) | r = 0.86 |
| θ (ability — drives scores) | r = 0.92 |
| b on low-N routed forms (N≈22) | r = 0.96 |
| c (guessing) | RMSE = 0.087 (low r is expected: narrow true-c range) |

## Validation 2 — benchmark vs **Xcalibre** (`benchmark_xcalibre.py`)

Ground-truth Xcalibre `a,b,c` (from `115.json.txt` + `A, B, C Values.xlsx`)
compared to our engine on the three overlapping forms (item ids aligned exactly
for 115 via the JSON; English aligned positionally for 117/119):

| Form / subject | b corr | c RMSE | a corr | a RMSE (SL-linked) |
|---|---|---|---|---|
| 115 RW | **0.961** | 0.022 | 0.60 | 0.17 |
| 115 Math | **0.960** | 0.034 | 0.17\* | 0.30 |
| 117 RW | **0.965** | 0.024 | 0.50 | 0.17 |
| 119 RW | **0.962** | 0.018 | 0.78 | 0.12 |

**Difficulty matches Xcalibre at r≈0.96 on every form; guessing matches to
≈0.02.** The `a` agreement tracks item-difficulty spread (0.78 on well-spread
form 119 → 0.17 on 115 Math, which is saturated with p>0.9 items where
discrimination is mathematically unidentifiable — Xcalibre's own SEs there are
large). Two matched conventions were the key: **D=1.702** and a **strong
Beta(0.25) guessing prior** (which is why Xcalibre's `c` clusters at 0.25).

### Where this engine goes *beyond* Xcalibre
- **Transparent QA flags** instead of silently zeroing bad items. On these four
  forms: 143 `too_easy`, 71 `low_N`, 2 `b_at_bound`, 1 `too_hard`.
- **Built-in linking** (`irt_engine/linking.py`: mean/mean, mean/sigma,
  Stocking–Lord) to place any calibration on a reference metric — used above to
  match Xcalibre, and the mechanism for unifying all four forms.
- **Provider-accurate scaling** read straight from the assessment JSON
  (`english_xbar/sigma`, `math_xbar/sigma`).

## Results on the four supplied forms

- 544 items calibrated, 1,938 student-form records scored.
- Marginal reliability 0.75–0.87 (appropriate for module-length sections).
- IRT-total vs raw-Total-Score correlation 0.86–0.94 — the intended divergence:
  IRT credits students routed to the **harder** Module 2 for facing harder
  items, which raw counting cannot.

## Run

```bash
python validate_recovery.py      # prove the engine (simulation)
python run_calibration.py        # calibrate + score the CSVs in this folder
```

Outputs land in `./output/`:
- `item_parameters.csv` — a, b, c, N, p-value, section per item
- `student_scores.csv` — section scores, total, θ, CSEM per student
- `calibration_summary.txt`

## Correction factor (matching Xcalibre's metric)

IRT parameters are identified only up to a linear transform of θ. Our calibration
lands on a slightly tighter metric than Xcalibre (θ scale ≈ 1.18× narrower for
RW). `benchmark_all.py` derives a per-subject **correction factor** (A, B) by
Stocking–Lord linking on the forms where we have Xcalibre truth (115/117/119),
pools them, and saves `output/correction_factors.json`:

```
reading_and_writing : A=1.175  B=-0.053
math                : A=1.005  B=-0.221
```

Applied as `a* = a/A`, `b* = A·b + B`, this makes our parameters numerically
interchangeable with Xcalibre's. Validated **out of sample**: on mocks
116/118/120 (not used to fit the factor) the corrected `b`/`a`/`c` distributions
match Xcalibre's to within ~0.05–0.2. It removes the metric offset; it cannot
(and should not) manufacture agreement on `a` for near-perfect-score items where
discrimination is genuinely unidentifiable.

## Scoring (θ → scaled score, with your own mean & SD)

```bash
python score_mock.py --mock 115 \
     --rw-mean 650 --rw-sd 70  --math-mean 700 --math-sd 70
```

Pipeline: calibrate → apply correction factor → EAP θ → `scaled = clip(mean +
SD·θ, 200, 800)` rounded to 10 → total = RW + Math. Mean/SD come from the CLI,
else `score_config.json` (keyed by mock), else the provider default (650/70,
700/70). Writes `output/scores_<mock>.csv` and saves the calibrated+corrected
item bank to `output/params_<mock>.json`.

### How future scoring happens
- **New mock (new items):** run `score_mock.py` once. It calibrates the bank
  from that mock's responses, corrects the metric, and scores everyone. The bank
  is saved to `params_<mock>.json`.
- **New students on an existing mock:** no recalibration. `score_students_fixed()`
  loads the saved bank and EAP-scores the new responses instantly
  (fixed-parameter scoring) — the θ metric and scale stay fixed, so a given
  response pattern always yields the same score. This is the routine path once a
  mock is live, and it is what makes scores stable and comparable over time.

## Known limitations & roadmap

1. **Cross-form comparability.** Each form is currently calibrated and scaled on
   its own cohort (norm-referenced to mean 500 / SD 100 per section), so a 600
   on form 115 is not yet strictly equal to a 600 on form 117. There is **zero
   common-item overlap** across forms but **102–261 shared students** between
   pairs → the immediate next step is **common-person (non-equivalent groups)
   equating** to place all four banks on one θ scale. *This is the single
   biggest upgrade and something a stock Xcalibre workflow does not give you.*
2. **Grid-in items.** Item-format metadata (MC vs student-produced-response)
   was not in the export, so grid-in `c` is currently learned from data via the
   shrinkage prior rather than fixed to 0. Feeding a format flag would let us
   fix `c=0` exactly for grid-ins (`ItemSpec(fit_c=False, fixed_c=0.0)`).
3. **Official raw→scale table.** Scaling is transparent norm-referenced linear;
   drop a College Board conversion table into `ScaleConfig` for exact-metric
   scores.
4. **Retakes / duplicates.** A few students appear multiple times per form
   (legitimate distinct sessions). Decide a policy (latest / best attempt)
   before reporting official scores.

## Package layout

```
irt_engine/
  model.py    MMLECalibrator (MMLE-EM 3PL + priors), Priors, ItemSpec
  score.py    eap_scores (EAP θ + CSEM), ScaleConfig (θ → 200–800)
  loader.py   Digital SAT CSV parser → per-subject response matrices
validate_recovery.py   simulation-based correctness proof
run_calibration.py     end-to-end pipeline over the folder
```
