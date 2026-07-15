"""
Benchmark every available mock against Xcalibre, and derive a generalizable
"correction factor" (linear metric transform) per subject.

Alignment:
  RW/English : positional via 'A, B, C Values.xlsx' English sheets (test order)
  Math       : only mock 115 (item ids via 115.json.txt); other Math sheets are
               reordered with no ids -> not comparable.

For each form/subject we compute the Stocking-Lord constants (A, B) that map our
metric onto Xcalibre's. Pooling those across forms gives a fixed correction we
can apply to *future* mocks that have no Xcalibre reference.
"""
import json, glob, re, numpy as np, pandas as pd
from irt_engine import MMLECalibrator, Priors
from irt_engine.loader import load_form
from irt_engine.linking import stocking_lord, apply_transform

XL = pd.ExcelFile("A, B, C Values.xlsx")
CSVS = {re.search(r"-\s*_?(\d{3})_", f).group(1): f
        for f in glob.glob("Digital_SAT_Mock_Test_*.csv")}


def xcal(form, subject, item_ids):
    if form == "115":
        q = pd.DataFrame(json.load(open("115.json.txt", encoding="utf-8"))["questions"])
        q["item_id"] = q["question"].astype(str)
        q["subject"] = np.where(q["name"].str.contains("Reading"),
                                "reading_and_writing", "math")
        m = q[q.subject == subject].set_index("item_id")
        idx = [str(i) for i in item_ids]
        return (np.array([m.loc[i].irt_a for i in idx]),
                np.array([m.loc[i].irt_b for i in idx]),
                np.array([m.loc[i].irt_c for i in idx]))
    if subject == "reading_and_writing":
        sh = XL.parse(f"{form} English")
        if len(sh) == len(item_ids):
            return sh["a"].values, sh["b"].values, sh["c"].values
    return None


def main():
    consts = {"reading_and_writing": [], "math": []}
    rows = []
    for form in sorted(CSVS):
        if form not in {"115", "116", "117", "118", "119", "120"}:
            continue
        for subject, fd in load_form(CSVS[form]).items():
            res = MMLECalibrator(max_iter=300, tol=5e-4,
                                 priors=Priors()).fit(fd.responses,
                                                      item_ids=fd.item_ids)
            ref = xcal(form, subject, res.item_ids)
            if ref is None:
                continue
            xa, xb, xc = ref
            keep = ~((xa == 0) & (xb == 0))
            oa, ob, oc = res.a[keep], res.b[keep], res.c[keep]
            xa, xb, xc = xa[keep], xb[keep], xc[keep]
            A, B = stocking_lord(oa, ob, oc, xa, xb, xc)
            la, lb, _ = apply_transform(oa, ob, oc, A, B)
            consts[subject].append((A, B))
            rows.append(dict(form=form, subject=subject, n=keep.sum(),
                             A=A, B=B,
                             b_corr=np.corrcoef(ob, xb)[0, 1],
                             a_corr=np.corrcoef(oa, xa)[0, 1],
                             b_rmse_linked=np.sqrt(np.mean((lb - xb) ** 2)),
                             c_within03=np.mean(np.abs(oc - xc) <= 0.03) * 100))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== Per-form agreement vs Xcalibre ===")
    print(df.round(3).to_string(index=False))

    print("\n=== Pooled CORRECTION FACTOR per subject (mean of A,B) ===")
    corr = {}
    for subj, lst in consts.items():
        if not lst:
            continue
        A = float(np.mean([x[0] for x in lst]))
        B = float(np.mean([x[1] for x in lst]))
        corr[subj] = (A, B)
        print(f"  {subj:20s} A={A:.4f}  B={B:+.4f}   (from {len(lst)} form(s))")
    json.dump(corr, open("output/correction_factors.json", "w"), indent=2)
    print("  -> saved output/correction_factors.json")

    # Apply pooled correction and re-measure b agreement (RW, all forms)
    print("\n=== Match after applying the POOLED correction (RW) ===")
    for r in rows:
        if r["subject"] != "reading_and_writing":
            continue
        A, B = corr["reading_and_writing"]
        # recompute using pooled A,B instead of per-form
        fd = load_form(CSVS[r["form"]])["reading_and_writing"]
        res = MMLECalibrator(max_iter=300, tol=5e-4, priors=Priors()).fit(
            fd.responses, item_ids=fd.item_ids)
        ref = xcal(r["form"], "reading_and_writing", res.item_ids)
        xa, xb, xc = ref
        keep = ~((xa == 0) & (xb == 0))
        la, lb, _ = apply_transform(res.a[keep], res.b[keep], res.c[keep], A, B)
        brmse = np.sqrt(np.mean((lb - xb[keep]) ** 2))
        b05 = np.mean(np.abs(lb - xb[keep]) <= 0.5) * 100
        print(f"  {r['form']} RW: b_rmse(pooled)={brmse:.3f}  within0.5={b05:.0f}%")


if __name__ == "__main__":
    main()
