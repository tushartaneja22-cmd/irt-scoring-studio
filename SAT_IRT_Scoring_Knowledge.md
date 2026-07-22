# How College Board Uses Item Response Theory (IRT) to Score the SAT

*A comprehensive, sourced reference on the psychometric machinery behind the Digital SAT Suite.*

**Last compiled:** July 2026
**Primary source:** College Board, *Digital SAT Suite of Assessments Technical Manual* (2024), Chapters 4–6 — the authoritative, first-party account. Public College Board pages and secondary explainers are used only to corroborate. Direct quotations and equation numbers below refer to that manual unless noted.

> **TL;DR.** The Digital SAT is a **two-stage multistage adaptive test (MST)** scored with **Item Response Theory**. Items are calibrated with the **3-parameter logistic (3PL) model** (and a **graded response model / 2PL** for Math student-produced responses). Your answer *pattern* plus each item's difficulty/discrimination/guessing parameters produce an ability estimate **θ (theta)**. On-device, an **EAP** estimate routes you to an easier or harder second module; after testing, a **Maximum Likelihood Estimate (MLE)** of θ is computed and mapped through a fixed **θ → scale-score conversion table** (built by equipercentile linking to the old paper SAT in 2022) to your 200–800 section scores, which sum to the 400–1600 total. Because scoring is pattern-based, **two students with the same number correct can get different scores.**

---

## Table of Contents
1. [The big picture: why IRT and why adaptive](#1-the-big-picture)
2. [Test structure: the 1–2 multistage adaptive design](#2-test-structure)
3. [The IRT models: 3PL and the graded response model](#3-the-irt-models)
4. [Ability estimation: EAP for routing, MLE for the final score](#4-ability-estimation)
5. [Routing: how the second module is chosen](#5-routing)
6. [Scaling: turning θ into a 200–800 score](#6-scaling)
7. [Equating: keeping every form comparable (pre-equating)](#7-equating)
8. [Item calibration & the item pool](#8-item-calibration)
9. [Reliability, information, and standard error](#9-reliability)
10. [Fairness: DIF, routing accuracy, and the "area of indifference"](#10-fairness)
11. [Content-domain subscores](#11-content-domain-subscores)
12. [Practical implications for test-takers](#12-practical-implications)
13. [Glossary](#13-glossary)
14. [Sources](#14-sources)

---

## 1. The big picture

**Item Response Theory (IRT)** is a family of statistical models that describe the probability a test-taker answers a given item correctly as a function of (a) the test-taker's latent ability, denoted **θ (theta)**, and (b) properties of the item itself (difficulty, discrimination, and guessability). College Board adopted an IRT-based, adaptive design for the Digital SAT (launched internationally March 2023, in the U.S. March 2024) because it "allows for precise measurement of students' knowledge and skills with fewer questions in less time than possible with traditional paper-and-pencil tests."

Two consequences flow directly from IRT and define modern SAT scoring:

- **Scores are not raw counts.** "Scoring is a function of a student's responses (i.e., based on the pattern of right and wrong answers) *and* the IRT characteristics (i.e., item statistics) of the set of items administered." A hard item answered correctly is stronger evidence of high ability than an easy one.
- **Equal number-correct ≠ equal score.** "Two students may have the same number of correct answers but have different reported scale scores," because the *difficulty of the specific items* they saw (especially which second module they were routed to) feeds the score.

IRT is the same broad methodology used by other major assessments cited in the manual and on College Board's site (NAEP, MAP, SBAC, PARCC).

---

## 2. Test structure

### 2.1 The "1–2" MST design
The SAT Suite uses **multistage testing (MST)** — described by College Board as "a middle ground between traditional linear-based tests and pure item-level adaptive, or computer adaptive (CAT), tests." Each **section** (Reading and Writing; Math) is an independent **panel** with **two stages**:

```
                         ┌─────────────────────────┐
                         │  STAGE 2 — Higher-       │
                    ┌───▶│  Difficulty Module       │
                    │    │  (items 26–50 / 21–44)   │
 ┌───────────────┐  │    └─────────────────────────┘
 │ STAGE 1        │  │  Routing
 │ Routing Module │──┤  decision
 │ broad mix of   │  │
 │ Easy/Med/Hard  │  │    ┌─────────────────────────┐
 └───────────────┘  └───▶│  STAGE 2 — Lower-        │
                         │  Difficulty Module       │
                         │  (items 26–50 / 21–44)   │
                         └─────────────────────────┘
```

- **Stage 1 = the routing module**: a single module spanning "a broad span of item difficulty," designed to measure a wide ability range.
- **Stage 2 = two candidate modules**: one lower-difficulty and one higher-difficulty. You take exactly **one**, chosen by the routing decision.
- Terminology (manual §6.1): *Item Pool → Module → Stage → Panel → Route.* A **panel** is the full two-stage section; a **route** is the path taken through it.

### 2.2 Counts and timing
| Section | Modules | Operational items / module | Pretest items / module | Time / module |
|---|---|---|---|---|
| Reading & Writing | 2 (Stage 1 + Stage 2) | 25 (≈27 total with pretest) | 2 | 32 min |
| Math | 2 (Stage 1 + Stage 2) | 20 (≈22 total with pretest) | 2 | 35 min |

- "Each stage consists of 50% of the total items that contribute to the student's score, plus two embedded pretest items in each stage of each section." So **4 pretest items per section**, **not scored**.
- RW is 54 items total (50 scored + 4 pretest); Math is 44 total (40 scored + 4 pretest).

### 2.3 Sections are scored independently
Routing and scoring for Reading & Writing are **completely separate** from Math. Your RW performance never affects your Math module, and vice versa.

### 2.4 Within-module review
Unlike a pure CAT, MST lets you **freely navigate and change answers within the current module** — but you **cannot return to a previous module** once the routing decision is made.

---

## 3. The IRT models

### 3.1 Three-parameter logistic (3PL) — the workhorse
For selected-response (multiple-choice) items, College Board uses the **3PL model** (Lord, 1980). The probability that test-taker *j* with ability **θⱼ** answers item *i* correctly is:

$$P_i(\theta_j) = c_i + (1 - c_i)\cdot \frac{1}{1 + \exp[-D\,a_i(\theta_j - b_i)]}$$

Parameters (manual §6.2.1):
- **aᵢ — discrimination.** How sharply the item distinguishes higher- from lower-ability students (the slope of the curve).
- **bᵢ — difficulty.** The θ location of the item; higher *b* = harder.
- **cᵢ — pseudo-guessing / lower asymptote.** "The probability that a student lacking complete knowledge would answer the item correctly." The curve's floor.
- **D — scaling constant**, set to **unity (D = 1)** for the SAT Suite. (Many texts use D = 1.702 to approximate the normal ogive; College Board explicitly uses 1.)

This is the same model family used by the **paper SAT** before the digital transition, easing the concordance between the two.

### 3.2 Graded response model (GRM ≈ 2PL) — for Math grid-ins
Math **student-produced-response (SPR)** items are calibrated with **Samejima's (1969) Graded Response Model**. When an item is scored dichotomously (right/wrong, as SAT SPRs are), the GRM **reduces to the 2-parameter logistic (2PL)** — i.e., the 3PL with **c = 0** (no guessing floor, appropriate because you can't "guess" a free-response numeric answer):

$$P_i(\theta_j) = \frac{1}{1 + \exp[-a_i(\theta_j - b_i)]}$$

College Board notes the distinction between the dichotomous GRM and the 2PL "is minor," and the GRM is the model actually used in estimation/calibration for SPR items in their software.

### 3.3 Pattern scoring
"The IRT models rely on **pattern scoring**, which incorporates all the parameters of each item administered to the examinee and whether the examinee answered the item correctly or incorrectly." This is the mathematical reason the SAT is *not* number-right scored.

---

## 4. Ability estimation

The SAT computes **θ twice** per adaptive section, using two different estimators for two different jobs.

### 4.1 EAP — Expected A Posteriori (on-device, for routing)
The **first** estimate, made at the end of Stage 1 to decide routing, is an **EAP** estimate:

$$\text{EAP}(\theta) = \frac{\sum_q \theta_q\,\big[\prod_i P_i^{u_i}(1-P_i)^{1-u_i}\big]\,W_q}{\sum_q \big[\prod_i P_i^{u_i}(1-P_i)^{1-u_i}\big]\,W_q}$$

where *uᵢ* = 1/0 for correct/incorrect, **θq** is the *q*-th quadrature point of a prior distribution, and **Wq** its weight.

- **Prior for routing:** a **uniform distribution on [−5, 5] with 101 quadrature points** — a deliberately *non-informative* prior (so Wq is constant and adds little assumption about the ability distribution).
- **Why EAP here:** it (1) can incorporate prior information, (2) avoids the optimization/convergence failures MLE can hit on short or all-correct/all-wrong response sets, and (3) handles sparse data well. Crucially it is **light enough to run on the student's device without internet**, so routing works offline inside the Bluebook testing app.
- EAP for routing uses **only the routing-module items**.

### 4.2 MLE — Maximum Likelihood Estimation (server-side, for the final score)
The **second** estimate, computed **after** the session when data returns to College Board, is the **MLE** — the θ that maximizes the likelihood of the observed response pattern across **all** scored items in the section:

$$L(\theta_j) = \prod_{i=1}^{n} P_i(\theta_j)^{u_i}\,[1 - P_i(\theta_j)]^{1-u_i}$$

- Solved numerically via the **Newton–Raphson** method.
- **The MLE θ is the basis for the student's final reported score** — for both adaptive and linear versions of the test.

**Summary:** EAP routes you (fast, on-device, uniform prior, routing module only). MLE scores you (server-side, full response pattern, feeds the scale-score table).

---

## 5. Routing

### 5.1 How the cut point is set
College Board uses a **population-distribution approach** to routing: "historic archive data of past student performance" determines the **median student performance**, which sets the routing decision point (the θ threshold between lower- and higher-difficulty Stage 2). Panels are assembled to **maximize measurement precision at exactly this routing point**. This approach also gives better **item-exposure control** than alternatives.

Mechanically: your Stage-1 EAP θ is compared to the cut; above it you get the harder module, below it the easier module.

### 5.2 It's about performance, not a fixed "get X right"
There is **no published fixed number** like "answer 15 right to unlock the hard module." The decision is an IRT-weighted θ estimate — *which* items you got right (their difficulty/discrimination) matters, not just how many. Secondary explainers that quote a specific count are approximations, not the mechanism.

### 5.3 Routing quality (from College Board's simulation studies)
Simulations of 10,000+ simulees, replicated 100×:
- **Routing accuracy ≈ 93%** (RW 93.20%, Math 93.25%) agreement between observed and "true" route; **Cohen's κ = 0.86** both sections.
- **Routing consistency ≈ 90%** (RW 90.40%, Math 90.41%) agreement across two different panels; **κ = 0.81**.
- Nearly all misroutes (99.8–99.9%) fall in the **"region of indifference"** — the score band where either route yields essentially the same scale score, so a misroute doesn't change the reported score.

---

## 6. Scaling

### 6.1 What "scaling" produces
Reported section scores are a **conversion of the IRT θ to a scale score**. Because θ is continuous on **[−5, 5]**, the scale is a set of **θ ranges each mapping to one of 61 unique scale-score points** (200, 210, …, 800).

### 6.2 The 2022 straight-line concordance study
The scale was built (2022) so digital scores would carry the **same meaning** as legacy paper-SAT scores:
- **18,513** U.S. and international 11th/12th graders took **both** the digital SAT and a paper SAT within a month of each other.
- Three θ estimators were compared for linking — **MLE, EAP, and a Test Characteristic Curve (TCC)** estimate.
- **Regression linking** reproduced individuals well but biased the overall distribution. **TCC linking** produced too many score gaps. **EAP linking** worked but required non-performance information, so it was kept only for routing and domain scores.
- **Winner: equipercentile linking of the digital MLE θ to the paper-SAT scale.** This produced the final **θ → scale-score conversion tables** for RW and Math.

Scaling design goals included: equal SAT means for the concordance sample; similar SDs/skewness to the paper SAT; SEMs similar to the paper test; **all-correct → 800**, **none-correct → 200**; and minimized gaps in the old scale.

### 6.3 The approximate θ → scale relationship
College Board publishes a polynomial *approximation* (their **Equation 5**) — used to estimate reliability and illustrate the relationship, **not** to compute the reported score (that comes from the lookup table):

$$\text{ScaleScore} \approx \text{Intercept} + \beta_1\theta + \beta_2\theta^2 + \beta_3\theta^3 + \beta_4\theta^4 + \beta_5\theta^5 + \beta_6\theta^6 + \beta_7\theta^7 + \beta_8\theta^8$$

### 6.4 Total score
$$\text{Total} = \text{RW section score} + \text{Math section score}$$
SAT total is **400–1600**; each section **200–800**.

### 6.5 Vertical scale across the Suite
The whole SAT Suite shares one **vertical scale**, with staggered floors/ceilings so growth can be tracked across grades:
| Assessment | Total | Section |
|---|---|---|
| SAT | 400–1600 | 200–800 |
| PSAT/NMSQT & PSAT 10 | 320–1520 | 160–760 |
| PSAT 8/9 | 240–1440 | 120–720 |

The vertical scale was established with a separate 2022 study (26,000+ students, grades 9–11, randomly assigned across assessment levels).

---

## 7. Equating (pre-equating)

The Digital SAT is **pre-equated** — "equating after an administration is no longer performed; instead all equating is done through pretesting calibrations and linking to a calibrated item pool and ATA assembly."

How comparability is guaranteed *before* anyone tests:
1. **Common calibrated pool.** Every item is calibrated onto one **common IRT metric** ("common-item equating to a calibrated pool," Kolen & Brennan, 2004). Because all items live on the same θ metric, any valid set of items yields equated θ's.
2. **Automated Test Assembly (ATA).** Linear-programming software selects items so every assembled panel meets identical content constraints and matching statistical targets — specifically matched **Test Characteristic Curves (TCCs)** and **Test Information Functions (TIFs)**.
   - **TCC constraint** ≈ traditional equating (matches expected number-correct across forms).
   - **TIF constraint** ≈ matched reliability/standard error across forms.
3. **Panel approval simulation.** Each assembled panel is simulated thousands of times; it is only approved if estimated **section reliability ≥ 0.90**, routing accuracy is appropriate, and its score distribution matches other panels. The result: "It would be a matter of indifference to students regarding which panel they are assigned."
4. **Exposure control.** Target maximum exposure for any child item is **5%**.

---

## 8. Item calibration

### 8.1 Embedded pretesting
Since October 2023, new items are **embedded** as the 4 unscored pretest items per section (2 per stage). Pretest items are randomly positioned (never the first/last two of a stage), the same across both Stage-2 difficulty levels so they're seen across the full ability range, and — for security — administered only to **domestic U.S. test-takers**. International test-takers instead see **anchor items** in those slots (existing calibrated items that tie data to the pool; also unscored).

### 8.2 Sample-size minimums
- **≥ 1,000 examinees** per pretest item, minimum.
- Items from the same **parent model** combine to **≥ 5,000 examinees**.

*("Parent model" = an item template/family; the Digital SAT generates multiple sibling "child items" from one parent, and they are analyzed together.)*

### 8.3 Classical flags (item screening)
A pretest item is flagged for review if:
- Item difficulty (proportion correct) **> 0.90** or **< 0.20**
- Item discrimination (item–total correlation) **< 0.20**
- Any distractor correlates with total score **> 0.05**

If any child item is problematic, the **entire parent model is flagged "do not use."**

### 8.4 IRT calibration with flexMIRT
- Software: **flexMIRT®** (Cai, 2017), using **Marginal Maximum Likelihood (MML)** to estimate item parameters.
- **Priors/hyper-priors** are placed on ability (normal), the slope (discrimination), the intercept (difficulty-related), and the lower-asymptote to aid convergence. SPR items use GRM (no lower-asymptote prior needed).
- **Fixed-item calibration:** operational items already on the metric are held fixed as **anchors**, placing the new items onto the **common IRT metric**. Anchor items are first vetted via **anchor item evaluation** (analogous to checking linking items in classical equating).
- **Convergence checks:** first- and second-order convergence per flexMIRT; item-fit statistics and item-fit plots reviewed.
- **IRT p-value consistency:** within a parent model, the sample-weighted IRT p-values must all fall within **0.10** of each other; otherwise the whole parent model is returned for revision/re-pretesting.
- Child items confirmed to arise from the same parent are **aggregated into a single item** for final calibration.

### 8.5 Ongoing quality management
After operational use, College Board continuously monitors: classical + IRT item analysis, **IRT parameter/scale drift** over time (conditional on ability), model fit, reliability/SEM, routing decisions, and empirical score distributions. Drifting items are flagged for recalibration or removal from the pool.

---

## 9. Reliability and error of measurement

### 9.1 Test Information Function (TIF)
Each item contributes **information** (precision) at a given θ; the section's **TIF** is the sum of item information functions. More information = smaller error. Panels are assembled to hit a **target TIF** (which is what pins reliability across forms).

### 9.2 Conditional SEM (in θ, then in scale-score units)
The conditional standard error at a person's θ estimate:

$$\text{cSEM}(\hat\theta_j) = \frac{1}{\sqrt{\text{TIF}(\hat\theta_j)}}$$

Because information is invariant under monotonic transformation, it is mapped to the scale-score metric using the **slope** dSS/dθ of the θ→scale relationship (derivative of Equation 5):

$$\text{cSEM}(SS_j) = \text{cSEM}(\hat\theta_j)\cdot\left(\frac{dSS}{d\theta}\right)$$

- The **cSEM in θ is capped (truncated) at 1.75.**
- Precision is best in roughly the **−2.5 to +1 θ** range and degrades at the extremes.

### 9.3 Total-score cSEM
$$\text{cSEM}(TSS_j) = \sqrt{\text{cSEM}(RWSS_j)^2 + \text{cSEM}(MSS_j)^2}$$

### 9.4 Reliability
$$\rho_{SS} = 1 - \frac{\text{SEM}^2_{SS}}{\hat\sigma^2_{SS}}$$
Every operational panel must achieve **section reliability ≥ 0.90** to be approved. Essay reliability is handled separately via inter-rater agreement (percentage agreement, single-rater SEM, Cohen's κ, and weighted κ).

### 9.5 Score *ranges*
Because of measurement error, College Board reports a **score range** (derived from the SEM) around each point score — "how much a student's scores would likely vary if they took a different administration under identical conditions."

---

## 10. Fairness

### 10.1 Differential Item Functioning (DIF)
Before operational use, every surviving parent model is checked for **DIF** using the **Mantel–Haenszel** procedure and the **ETS A/B/C classification**:
- **A** = negligible DIF, **B** = moderate, **C** = strong.
- **Any parent model with C-level DIF for any subgroup is flagged "do not use."**
- Negative item-total correlations also trigger rejection.

### 10.2 Is the adaptive routing itself fair?
College Board's simulation ("Fairness of Adaptive Multistage Tests") generated **61,000 simulees** (1,000 at each scale score 200–800) and asked whether routing helps or hurts. Findings:
- The **"area of indifference"**: across most of the scale, either route yields essentially the same score.
- For **high-ability** students, being (mis)routed to the *easier* module would slightly *over*estimate their score — but to trigger that routing they'd have to deliberately miss routing-module items, which *caps* their maximum achievable score below what honest effort yields. Example: a strategic high-ability student who tanks routing to land on the easy Math module and then aces it tops out around **580**.
- **Takeaway (College Board's):** "A student has no reason to attempt to manipulate the system and should always strive to answer as many questions correctly as possible."

### 10.3 CCR benchmark classification accuracy
College and Career Readiness benchmarks — **RW = 480, Math = 530** — classify students as at/above vs. below. Simulated **classification accuracy > 94%** and **consistency > 92%** for both sections, adaptive and linear, with high κ.

### 10.4 Structural validity
**Confirmatory factor analysis** (per route) and **generalizability theory** support treating each section as **unidimensional** (one factor for Math, one for RW) — the precondition for a single-θ IRT model. Fit judged by RMSEA/SRMR (< 0.05 good) and CFI/TLI (> 0.95 good).

---

## 11. Content-domain subscores

Beyond the two section scores, each section reports performance in **four content domains** (eight total). These are:
- Estimated by **EAP**, but with a **normal prior** (mean + SD), because domains have very few items (as few as **4**, up to **18**).
- Reported as one of **seven performance categories** set by a **scale-anchoring** process (bands from the 25th percentile of PSAT 8/9 up to the top 10th percentile of SAT).
- **Explicitly low precision.** College Board cautions against over-interpreting domain categories or comparing students on them — with so few items, a lower-ability student can occasionally look stronger on a hard domain than a higher-ability one.

---

## 12. Practical implications for test-takers

1. **There's no "gaming" the adaptive routing.** Deliberately missing routing questions to get an easier module lowers your ceiling. Always do your best on every item.
2. **Getting the harder Stage-2 module is good news** — it means Stage-1 performance was strong, and the harder module allows access to the top of the scale for a given number correct.
3. **Number correct ≠ score.** Which items you got right — and especially which Stage-2 module you were routed to — shapes the scale score. Two students with the *same raw count correct* can receive different section scores if they took different modules; the harder module maps a given count to a higher scale score.
4. **You can review within a module** — use remaining module time to check work — **but you can't go back to an earlier module.**
5. **Section scores are independent.** A rough RW section doesn't drag down Math routing or scoring.
6. **Score ranges matter.** Treat a reported score as the center of a band (± the SEM), not a razor-exact value.
7. **Pretest/anchor items don't count.** Four items per section are experimental; you can't identify them, so treat every item as scored.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **θ (theta)** | Latent ability estimated for each section; SAT metric runs −5 to +5. |
| **3PL** | 3-parameter logistic IRT model: discrimination *a*, difficulty *b*, guessing *c*. |
| **2PL / GRM** | 2-parameter model (guessing = 0); GRM reduces to 2PL for dichotomous Math SPR items. |
| **a / b / c** | Discrimination / difficulty / pseudo-guessing (lower asymptote). |
| **EAP** | Expected A Posteriori θ estimate — used on-device for routing and for domain scores. |
| **MLE** | Maximum Likelihood θ estimate (Newton–Raphson) — basis of the final reported score. |
| **MST** | Multistage adaptive test; SAT uses a "1–2" design (1 routing module → 2 candidate modules). |
| **Panel / Module / Stage / Route** | Panel = full section; module = item set; stage = a routing decision level; route = the path taken. |
| **Routing module** | Stage-1 module with broad difficulty; determines Stage-2 assignment. |
| **Pre-equating** | Equating done *before* administration via calibrated pool + ATA (no post-hoc equating). |
| **ATA** | Automated Test Assembly — linear-programming form construction to content + statistical targets. |
| **TCC / TIF** | Test Characteristic Curve (expected number-correct vs θ) / Test Information Function (precision vs θ). |
| **cSEM** | Conditional standard error of measurement at a given θ / scale score. |
| **flexMIRT** | Software used for MML item calibration and θ estimation. |
| **MML** | Marginal Maximum Likelihood — estimates item parameters during calibration. |
| **Parent model / child item** | An item template and the sibling items generated from it; calibrated together. |
| **DIF** | Differential Item Functioning; SAT uses Mantel–Haenszel + ETS A/B/C classification. |
| **Anchor item** | A previously calibrated item used to place new items on the common metric (unscored). |
| **CCR benchmark** | College & Career Readiness cut: RW 480, Math 530. |
| **Scale-anchoring** | Method for setting the seven content-domain performance bands. |

---

## 14. Sources

**Primary (authoritative):**
- College Board. *Digital SAT Suite of Assessments Technical Manual* (2024). Especially **Chapter 4** (Multistage Adaptive Testing, Embedded Pretesting), **Chapter 5** (Test Scoring and Reporting), and **Chapter 6** (Psychometrics: §6.1 Adaptive Testing, §6.2 Scaling and Norming / IRT, §6.3 Reliability and Errors of Measurement, §6.4 Item Analysis, Calibrations, and Pre-Equating, §6.5 Panel Assembly and Ongoing Psychometric Quality Management). PDF: https://research.collegeboard.org/media/pdf/Digital%20SAT%20Suite%20of%20Assessments%20Technical%20Manual-FINAL.pdf

**Corroborating (College Board public):**
- College Board — [How Are Scores Calculated?](https://satsuite.collegeboard.org/scores/what-scores-mean/how-scores-calculated)
- College Board Blog — [What Is Digital SAT Adaptive Testing?](https://blog.collegeboard.org/what-digital-sat-adaptive-testing)

**Secondary explainers (context only; not authoritative for parameters):**
- Compass Education Group — [The New Digital SAT](https://www.compassprep.com/digital-sat/)
- Applerouth — [Deep Dive on the Digital SAT Practice Tests](https://www.applerouth.com/blog/deep-dive-on-the-4-digital-sat-practice-tests)

**Foundational IRT literature referenced by the manual:**
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems.* (3PL model.)
- Samejima, F. (1969). *Estimation of latent ability using a response pattern of graded scores.* (Graded Response Model.)
- Cai, L. (2017). *flexMIRT® v3.5* user's manual; Houts & Cai (2016) flexMIRT documentation.
- Kolen, M. J., & Brennan, R. L. (2004). *Test Equating, Scaling, and Linking.* (Common-item equating to a calibrated pool.)

---

*Compiled for the IRT Scoring Studio project. All parameter definitions, equations, thresholds (e.g., cSEM cap of 1.75, reliability ≥ 0.90, exposure ≤ 5%, pretest flag cutoffs, DIF handling) are drawn directly from College Board's 2024 Technical Manual. Where the manual and secondary sources conflict, the manual governs.*
