"""LaTeX for the paper's evidence-ambiguity table.

Each scored GT point is bucketed by how many distinct classes deposited evidence on the primitive
that OWNS it. One class means the argmax is determined before any solver runs -- there is no
decision to make -- so that bucket is the part of the score a lift gets for free. Two or more means
the weighting has to adjudicate a genuine conflict.

Accuracy is decomposable over these buckets and recomposes the reported number exactly. mIoU is NOT:
it averages over the classes present in whatever point set it is given, so each bucket's mIoU uses
its own denominator and the buckets do not recombine. The column is therefore labelled as
"mIoU if only these points were scored" and the per-bucket class count is printed in the audit so a
bucket whose denominator collapsed is visible rather than silently flattering.
"""
from __future__ import annotations
import argparse
import json
import statistics as st

ARMS = [("truefrozen", r"PowerFoam (frozen)"), ("nonfrozen", r"PowerFoam (unfrozen)"),
        ("gs_froz", r"3DGS (frozen)"), ("gs_unfroz", r"3DGS (unfrozen)")]
BUCKETS = [("ev1", r"$1$"), ("ev2", r"$2$"), ("ev3p", r"$\geq 3$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/scannet/oracle_buckets.json")
    ap.add_argument("--solver", default="closed")
    ap.add_argument("--out", default="artifacts/tab_buckets.tex")
    ap.add_argument("--caption", default=None,
                    help="file holding the table environment, with %%TABULAR%% where the tabular goes")
    ap.add_argument("--also", default=None, help="comma-separated extra paths to write the same tex")
    a = ap.parse_args()
    d = json.load(open(a.rows))
    scenes = sorted({r["scene"] for r in d})

    have_miou = any(f"b_ev1_miou_{a.solver}" in r for r in d)
    L = []
    L.append(r"\begin{tabular}{llrrr}")
    L.append(r"\toprule")
    L.append(r"& evidence classes & \% of points & Acc. & "
             + (r"mIoU$^\dagger$ \\" if have_miou else r"\\"))
    L.append(r"\midrule")
    for ai, (arm, lbl) in enumerate(ARMS):
        s = [r for r in d if r["recon"] == arm]
        if not s:
            continue
        for bi, (b, blab) in enumerate(BUCKETS):
            m = lambda k: st.mean([r[k] for r in s if k in r])
            fr = m(f"b_{b}_frac_{a.solver}") * 100
            # Weighted so the table RECOMPOSES: with acc_b = mean_s(frac_bs*acc_bs)/mean_s(frac_bs),
            # sum_b mean_s(frac_bs) * acc_b == mean_s(acc_s), the reported per-point accuracy, to
            # float32. Weighting by n_scored instead pools by raw points and does NOT reconstruct
            # the headline (which is an unweighted scene mean), so the columns would silently not
            # add up. The audit below checks the identity on every run.
            num = st.mean([r[f"b_{b}_frac_{a.solver}"] * r[f"b_{b}_acc_{a.solver}"] for r in s])
            ac = 100 * num / (fr / 100)
            mnum = st.mean([r[f"b_{b}_frac_{a.solver}"] * r.get(f"b_{b}_miou_{a.solver}", 0.0)
                            for r in s])
            head = lbl if bi == 0 else ""
            cells = f"{head} & {blab} & {fr:.1f} & {ac:.2f}"
            if have_miou:
                cells += f" & {100 * mnum / (fr / 100):.2f}"
            L.append(cells + r" \\")
        if ai < len(ARMS) - 1:
            L.append(r"\addlinespace")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    tex = "\n".join(L)
    if a.caption:
        # wrap the generated tabular in the paper caption so the FILE THE PAPER INPUTS is
        # generated, not hand-maintained -- hand-copied cells drift silently from the data.
        tex = open(a.caption, encoding="utf-8").read().rstrip().replace("%%TABULAR%%", tex)
    open(a.out, "w", encoding="utf-8").write(tex + chr(10))
    for extra in [x for x in (a.also or "").split(",") if x]:
        open(extra, "w", encoding="utf-8").write(tex + chr(10))
        print("% also wrote " + extra)
    print(tex)

    print(f"\n% --- audit ({len(scenes)} scenes) ---")
    for arm, lbl in ARMS:
        s = [r for r in d if r["recon"] == arm]
        if not s:
            print(f"%  {lbl}: NO ROWS"); continue
        m = lambda k: st.mean([r[k] for r in s if k in r])
        tot = sum(m(f"b_{b}_frac_{a.solver}") for b, _ in BUCKETS)
        # per-scene decomposition is exact; average the per-scene sums, never the factors
        per = [sum(r[f"b_{b}_frac_{a.solver}"] * r[f"b_{b}_acc_{a.solver}"] for b, _ in BUCKETS)
               for r in s]
        rec = st.mean(per)
        disp = sum(st.mean([r[f"b_{b}_frac_{a.solver}"] for r in s]) *
                   (st.mean([r[f"b_{b}_frac_{a.solver}"] * r[f"b_{b}_acc_{a.solver}"] for r in s]) /
                    st.mean([r[f"b_{b}_frac_{a.solver}"] for r in s])) for b, _ in BUCKETS)
        act = m(f"pt_acc_centre_{a.solver}")
        flag = ("OK" if abs(tot - 1) < 1e-6 and abs(rec - act) < 5e-4 and abs(disp - act) < 5e-4
                else "*** MISMATCH ***")
        line = (f"%  {lbl:<22} n={len(s):<3} sum(frac)={tot:.6f} "
                f"recomposed acc={rec * 100:.3f} vs actual={act * 100:.3f}  {flag}")
        if have_miou:
            ncls = [f"{b}:{st.mean([r[f'b_{b}_ncls_{a.solver}'] for r in s if f'b_{b}_ncls_{a.solver}' in r]):.1f}"
                    for b, _ in BUCKETS]
            line += "  classes/bucket " + " ".join(ncls)
        print(line)
    if not have_miou:
        print("%  NOTE: no per-bucket mIoU in this file -- rerun oracle_projected.py to populate it.")


if __name__ == "__main__":
    main()
