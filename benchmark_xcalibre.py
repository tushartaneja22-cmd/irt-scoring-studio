"""
Benchmark the upgraded engine against Xcalibre ground truth.

Form 115: exact item-id alignment via 115.json.txt (RW + Math).
Forms 117/119: English aligned positionally via 'A, B, C Values.xlsx'
               (Excel English rows follow test order; Math sheet is reordered
               and has no ids, so Math is validated on 115 only).

Reports Pearson r and RMSE for a, b, c -- overall and on the subset where
discrimination is identifiable (p<=0.9, excluding Xcalibre-dropped items) --
plus RMSE after Stocking-Lord linking onto Xcalibre's metric.
"""
import json
import numpy as np, pandas as pd
from irt_engine import MMLECalibrator, Priors
from irt_engine.loader import load_form
from irt_engine.linking import stocking_lord, apply_transform

CSV = {
    "115": "Digital_SAT_Mock_Test_-115_1754312309.csv",
    "117": "Digital_SAT_Mock_Test_-117_1774194364.csv",
    "119": "Digital_SAT_Mock_Test_-119_1762971665.csv",
}


def calibrate(csv, subject):
    fd = load_form(csv)[subject]
    cal = MMLECalibrator(max_iter=300, tol=5e-4, priors=Priors())
    return fd, cal.fit(fd.responses, item_ids=fd.item_ids)


def xc_from_json(path):
    q = pd.DataFrame(json.load(open(path, encoding="utf-8"))["questions"])
    q["item_id"] = q["question"].astype(str)
    q["subject"] = np.where(q["name"].str.contains("Reading"),
                            "reading_and_writing", "math")
    return q


def report(name, oa, ob, oc, xa, xb, xc, D=1.702):
    keep = ~((xa == 0) & (xb == 0))                 # drop Xcalibre-zeroed
    oa, ob, oc, xa, xb, xc = [v[keep] for v in (oa, ob, oc, xa, xb, xc)]
    def cr(x, y): return np.corrcoef(x, y)[0, 1]
    def rmse(x, y): return np.sqrt(np.mean((x - y) ** 2))
    A, B = stocking_lord(oa, ob, oc, xa, xb, xc, D=D)
    la, lb, _ = apply_transform(oa, ob, oc, A, B)
    print(f"\n### {name}  (n={keep.sum()} after dropping "
          f"{(~keep).sum()} Xcalibre-dropped)   SL link A={A:.3f} B={B:+.3f}")
    print(f"  a: r={cr(oa,xa):.3f}  rmse={rmse(oa,xa):.3f} -> linked {rmse(la,xa):.3f}")
    print(f"  b: r={cr(ob,xb):.3f}  rmse={rmse(ob,xb):.3f} -> linked {rmse(lb,xb):.3f}")
    print(f"  c: r={cr(oc,xc):.3f}  rmse={rmse(oc,xc):.3f}")


def main():
    # ---- 115 : exact id alignment (RW + Math) ----
    xc = xc_from_json("115.json.txt")
    for subject in ["reading_and_writing", "math"]:
        fd, res = calibrate(CSV["115"], subject)
        xmap = xc[xc.subject == subject].set_index("item_id")
        idx = [str(i) for i in res.item_ids]
        xa = np.array([xmap.loc[i].irt_a for i in idx])
        xb = np.array([xmap.loc[i].irt_b for i in idx])
        xc_ = np.array([xmap.loc[i].irt_c for i in idx])
        report(f"115 {subject}", res.a, res.b, res.c, xa, xb, xc_, D=res.D)
        nflag = sum(f != "ok" for f in res.flags)
        print(f"  QA flags: {nflag}/{len(res.flags)} items flagged")

    # ---- 117 / 119 English : positional alignment via Excel ----
    xl = pd.ExcelFile("A, B, C Values.xlsx")
    for form in ["117", "119"]:
        fd, res = calibrate(CSV[form], "reading_and_writing")
        sheet = xl.parse(f"{form} English")
        if len(sheet) == len(res.a):
            report(f"{form} English (positional)", res.a, res.b, res.c,
                   sheet["a"].values, sheet["b"].values, sheet["c"].values,
                   D=res.D)


if __name__ == "__main__":
    main()
