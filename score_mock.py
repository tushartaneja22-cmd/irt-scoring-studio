"""
Score a mock: theta  ->  scaled section scores  ->  total.

Pipeline
  1. calibrate 3PL item parameters from the student responses (MMLE-EM + priors)
  2. apply the CORRECTION FACTOR (output/correction_factors.json) so ability is
     on Xcalibre's / the provider's metric
  3. EAP-score each student  (theta, CSEM)
  4. convert to the scaled score with a per-subject mean & SD that YOU enter:
        scaled = clip( mean + SD * theta , 200 , 800 )  rounded to 10
  5. total = RW + Math

Enter the mean/SD on the command line, e.g.:

  python score_mock.py --mock 115 \
       --rw-mean 650 --rw-sd 70  --math-mean 700 --math-sd 70

If a flag is omitted we fall back to score_config.json (keyed by mock number),
then to the provider default (RW 650/70, Math 700/70). The calibrated+corrected
item bank is saved to output/params_<mock>.json so future students can be scored
instantly WITHOUT recalibrating (see score_students_fixed()).
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from irt_engine import MMLECalibrator, Priors, eap_scores
from irt_engine.loader import load_form

OUT = "output"
PROVIDER_DEFAULT = {"reading_and_writing": (650.0, 70.0), "math": (700.0, 70.0)}


def load_corrections():
    p = os.path.join(OUT, "correction_factors.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def find_csv(mock):
    hits = [f for f in glob.glob("Digital_SAT_Mock_Test_*.csv")
            if re.search(rf"-\s*_?{mock}_", f)]
    if not hits:
        raise FileNotFoundError(f"no CSV for mock {mock}")
    return hits[0]


def scale(theta, mean, sd, lo=200, hi=800, step=10):
    raw = mean + sd * np.asarray(theta)
    return np.clip(np.round(raw / step) * step, lo, hi).astype(int)


def score_mock(mock, means_sds, apply_correction=True, save_bank=True):
    """means_sds: {subject: (mean, sd)}.  Returns a per-student DataFrame."""
    corr = load_corrections()
    forms = load_form(find_csv(mock))
    frame, bank = None, {"mock": mock, "subjects": {}}
    for subject, fd in forms.items():
        res = MMLECalibrator(max_iter=300, tol=5e-4,
                             priors=Priors()).fit(fd.responses,
                                                  item_ids=fd.item_ids)
        a, b, c = res.a.copy(), res.b.copy(), res.c.copy()
        A, B = corr.get(subject, [1.0, 0.0]) if apply_correction else (1.0, 0.0)
        # correct item params onto the reference metric
        a_c, b_c = a / A, A * b + B
        theta, se = eap_scores(fd.responses, a_c, b_c, c, D=res.D)
        mean, sd = means_sds.get(subject, PROVIDER_DEFAULT[subject])
        scaled = scale(theta, mean, sd)
        bank["subjects"][subject] = dict(
            item_ids=list(map(str, res.item_ids)),
            a=[round(float(x), 4) for x in a_c],
            b=[round(float(x), 4) for x in b_c],
            c=[round(float(x), 4) for x in c],
            D=res.D, scale_mean=mean, scale_sd=sd)
        df = pd.DataFrame({
            "student_id": fd.student_ids, "name": fd.names, "email": fd.emails,
            f"theta_{subject}": np.round(theta, 4),
            f"csem_{subject}": np.round(se, 4),
            f"score_{subject}": scaled})
        frame = df if frame is None else frame.merge(
            df, on=["student_id", "name", "email"], how="outer")

    rw, ma = frame.get("score_reading_and_writing"), frame.get("score_math")
    if rw is not None and ma is not None:
        frame["total_score"] = (rw.fillna(0) + ma.fillna(0)).astype(int)
    frame.insert(0, "mock", mock)
    if save_bank:
        json.dump(bank, open(os.path.join(OUT, f"params_{mock}.json"), "w"),
                  indent=2)
    return frame


def score_students_fixed(mock, responses_by_subject):
    """Fixed-parameter scoring of NEW students on an already-calibrated mock.
    responses_by_subject: {subject: (P,I) array aligned to saved item order}.
    No recalibration -- just EAP with the saved bank. This is how routine
    future scoring runs."""
    bank = json.load(open(os.path.join(OUT, f"params_{mock}.json")))
    out = {}
    for subject, R in responses_by_subject.items():
        s = bank["subjects"][subject]
        theta, se = eap_scores(R, np.array(s["a"]), np.array(s["b"]),
                               np.array(s["c"]), D=s["D"])
        out[subject] = dict(theta=theta, se=se,
                            score=scale(theta, s["scale_mean"], s["scale_sd"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True)
    ap.add_argument("--rw-mean", type=float); ap.add_argument("--rw-sd", type=float)
    ap.add_argument("--math-mean", type=float); ap.add_argument("--math-sd", type=float)
    ap.add_argument("--no-correction", action="store_true")
    a = ap.parse_args()

    cfg = {}
    p = "score_config.json"
    if os.path.exists(p):
        cfg = json.load(open(p)).get(a.mock, {})

    def pick(cli_m, cli_s, subj):
        d = PROVIDER_DEFAULT[subj]
        m = cli_m if cli_m is not None else cfg.get(subj, d)[0]
        s = cli_s if cli_s is not None else cfg.get(subj, d)[1]
        return (m, s)

    means_sds = {"reading_and_writing": pick(a.rw_mean, a.rw_sd, "reading_and_writing"),
                 "math": pick(a.math_mean, a.math_sd, "math")}
    print(f"Scoring mock {a.mock} with mean/SD:")
    for k, v in means_sds.items():
        print(f"  {k:20s} mean={v[0]:.0f} sd={v[1]:.0f}")
    df = score_mock(a.mock, means_sds, apply_correction=not a.no_correction)
    out = os.path.join(OUT, f"scores_{a.mock}.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} students -> {out}")
    print("saved item bank -> output/params_%s.json" % a.mock)
    print("\nScore distribution:")
    print(df[["score_reading_and_writing", "score_math", "total_score"]]
          .describe().round(0).to_string())


if __name__ == "__main__":
    main()
