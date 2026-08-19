"""Merge the per-scene semantic-surface results into one 10-scene table per method.

Each method's 10 scenes arrive in two pieces (scene0000_00 was run locally, the other nine
remotely), so this stitches them and reports the aggregate. Averaging is over scenes, and
each scene's own value was already averaged over the classes PRESENT in that scene --
matching the mIoU protocol's present-classes convention rather than averaging over a fixed
class list.
"""
import argparse
import json

import numpy as np

KEYS = ("scd", "mae_pred2gt", "mae_gt2pred", "hd95", "boundary_f1", "mIoU", "mAcc")
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]


def merge(paths):
    per = {}
    for p in paths:
        d = json.load(open(p))["per_scene"]
        for cs, scenes in d.items():
            per.setdefault(cs, {}).update(scenes)
    return per


def agg(per_cs):
    out = {k: float(np.mean([v[k] for v in per_cs.values()])) for k in KEYS}
    out["n"] = len(per_cs)
    out["missed"] = float(np.mean([v["n_missed"] for v in per_cs.values()]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foam", nargs="+", required=True)
    p.add_argument("--gaussian", nargs="+", required=True)
    p.add_argument("--label-foam", default="Feature Foam (champion stack)")
    p.add_argument("--label-gauss", default="Splat Feature Solver (per-primitive argmax)")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    foam, gauss = merge(a.foam), merge(a.gaussian)
    table, summary = [], {}
    for cs in CLASS_SETS:
        f, g = agg(foam[cs]), agg(gauss[cs])
        summary[cs] = {"foam": f, "gaussian": g}
        for name, m in ((a.label_foam, f), (a.label_gauss, g)):
            table.append((cs, name, m))

    hdr = (f"{'class set':<16}{'method':<42}{'n':>3} {'mIoU':>6} {'mAcc':>6} "
           f"{'semCD':>7} {'p->g':>7} {'g->p':>7} {'HD95':>8} {'bF1':>6}")
    print(hdr)
    print("-" * len(hdr))
    for cs, name, m in table:
        print(f"{cs:<16}{name:<42}{m['n']:>3} {m['mIoU']*100:>6.2f} {m['mAcc']*100:>6.2f} "
              f"{m['scd']*100:>7.2f} {m['mae_pred2gt']*100:>7.2f} {m['mae_gt2pred']*100:>7.2f} "
              f"{m['hd95']*100:>8.2f} {m['boundary_f1']:>6.3f}")

    print("\nper-scene semantic CD (cm), 19cls -- consistency check:")
    for scene in sorted(foam["opengaussian19"]):
        fv = foam["opengaussian19"][scene]["scd"] * 100
        gv = gauss["opengaussian19"].get(scene, {}).get("scd", float("nan")) * 100
        flag = "foam" if fv < gv else "GAUSS"
        print(f"  {scene}: foam {fv:6.2f}  gaussian {gv:6.2f}   -> {flag}")
    wins = sum(1 for s in foam["opengaussian19"]
               if s in gauss["opengaussian19"]
               and foam["opengaussian19"][s]["scd"] < gauss["opengaussian19"][s]["scd"])
    n = sum(1 for s in foam["opengaussian19"] if s in gauss["opengaussian19"])
    print(f"  foam wins {wins}/{n} scenes on semantic CD (19cls)")

    json.dump({"summary": summary, "foam_per_scene": foam, "gaussian_per_scene": gauss},
              open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
