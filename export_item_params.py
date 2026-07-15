"""
Per-mock item-parameter CSVs built from student response data.

For each mock (115, 117, 119) writes output/item_params_<form>.csv with our
3PL a,b,c (from the students' responses) plus the Xcalibre reference columns
and match diagnostics where alignment is possible:

  115         RW + Math aligned exactly by item id (115.json.txt)
  117 / 119   RW aligned positionally (A, B, C Values.xlsx English sheets);
              Math has no id-bearing reference -> our params only.

Our a,b are additionally reported on Xcalibre's metric (Stocking-Lord linked)
so the two are directly comparable; c needs no linking.
"""
import json, numpy as np, pandas as pd
from irt_engine import MMLECalibrator, Priors
from irt_engine.loader import load_form
from irt_engine.linking import stocking_lord, apply_transform

CSV = {"115": "Digital_SAT_Mock_Test_-115_1754312309.csv",
       "117": "Digital_SAT_Mock_Test_-117_1774194364.csv",
       "119": "Digital_SAT_Mock_Test_-119_1762971665.csv"}
XL = pd.ExcelFile("A, B, C Values.xlsx")


def xcalibre_for(form, subject, item_ids):
    """Return aligned (a,b,c) arrays or None if no reference available."""
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
    if subject == "reading_and_writing":                 # positional English
        sh = XL.parse(f"{form} English")
        if len(sh) == len(item_ids):
            return sh["a"].values, sh["b"].values, sh["c"].values
    return None


def main():
    for form, csv in CSV.items():
        rows = []
        for subject, fd in load_form(csv).items():
            res = MMLECalibrator(max_iter=300, tol=5e-4,
                                 priors=Priors()).fit(fd.responses,
                                                      item_ids=fd.item_ids)
            ref = xcalibre_for(form, subject, res.item_ids)
            if ref is not None:
                xa, xb, xc = ref
                keep = ~((xa == 0) & (xb == 0))          # exclude XC-dropped
                A, B = stocking_lord(res.a[keep], res.b[keep], res.c[keep],
                                     xa[keep], xb[keep], xc[keep])
                la, lb, _ = apply_transform(res.a, res.b, res.c, A, B)
            for i, iid in enumerate(res.item_ids):
                row = dict(form=form, subject=subject,
                           section=fd.item_sections[i], item_id=iid,
                           N=int(res.n_administered[i]),
                           p_value=round(float(res.p_correct[i]), 4),
                           a=round(float(res.a[i]), 4),
                           b=round(float(res.b[i]), 4),
                           c=round(float(res.c[i]), 4),
                           qa_flag=res.flags[i])
                if ref is not None:
                    xcdrop = (xa[i] == 0 and xb[i] == 0)
                    row.update(
                        a_linked=round(float(la[i]), 4),
                        b_linked=round(float(lb[i]), 4),
                        a_xcalibre=None if xcdrop else round(float(xa[i]), 4),
                        b_xcalibre=None if xcdrop else round(float(xb[i]), 4),
                        c_xcalibre=None if xcdrop else round(float(xc[i]), 4),
                        b_abs_diff=None if xcdrop else round(abs(lb[i] - xb[i]), 4),
                        c_abs_diff=None if xcdrop else round(abs(res.c[i] - xc[i]), 4),
                        xcalibre_dropped=bool(xcdrop))
                rows.append(row)
        df = pd.DataFrame(rows)
        out = f"output/item_params_{form}.csv"
        df.to_csv(out, index=False)
        # summary line
        haveref = "b_xcalibre" in df and df["b_xcalibre"].notna().any()
        msg = f"{out}: {len(df)} items"
        if haveref:
            v = df.dropna(subset=["b_xcalibre"])
            for subj in v.subject.unique():
                s = v[v.subject == subj]
                bcorr = np.corrcoef(s.b_linked, s.b_xcalibre)[0, 1]
                acorr = np.corrcoef(s.a_linked, s.a_xcalibre)[0, 1]
                b05 = (s.b_abs_diff <= 0.5).mean() * 100
                c03 = (s.c_abs_diff <= 0.03).mean() * 100
                msg += (f"\n    {subj:20s} vs Xcalibre: b_corr={bcorr:.2f} "
                        f"(within0.5={b05:.0f}%)  a_corr={acorr:.2f}  "
                        f"c_within0.03={c03:.0f}%  (n={len(s)})")
        print(msg)


if __name__ == "__main__":
    main()
