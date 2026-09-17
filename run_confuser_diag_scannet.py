"""Two-way confuser diagnosis on ScanNet, for foam AND 3DGS.

THE DECOMPOSITION. Two very different things produce a failing class:

  (i)  INFORMATION failure -- CLIP cannot separate the class from its confuser at all. Then even a
       2-WAY contest between just those two classes is near chance (or below it), and no lifting,
       decision rule or normalisation can recover what is not there.
  (ii) COMPETITION failure -- the evidence IS present, but in the K-way arg-max the class loses to a
       HUB that beats it everywhere. Then 2-way >> 50% and the failure is fixable.

The K-way accuracy we normally report cannot tell these apart. Per class c we compute:

  acc_K     accuracy in the real K-way task
  acc_2way  accuracy restricted to {c, its top confuser}, over cells whose GT is c. Chance = 50%.
  text_cos  cosine between the two class TEXT embeddings -- how close the words are before any
            image evidence enters.

WHY BOTH REPRESENTATIONS. On ScanNet++ this diagnosis found failing classes scoring 30.9% two-way --
BELOW chance -- with confusers that are hypernyms of their victims (kitchen cabinet -> cabinet, text
cos 0.829). If that reproduces on ScanNet for foam AND for 3DGS, the cause is the text/vocabulary
side and is representation-independent; if it appears in only one, it is a property of the lift.
Running both under one script keeps the class set, the text embeddings and the arg-max identical, so
the comparison is not confounded by protocol drift.

SCORING VOCABULARY. The arg-max runs over the FULL class set, not the per-scene present classes.
A confuser that never appears in a scene is exactly the kind of hub this diagnosis exists to find,
and restricting to present classes would hide it. This differs from the headline mIoU protocol and
the numbers here are therefore not comparable to it.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")

ARMS = {
    "foam": ("powerfoam", "output/scannet_{s}_truefrozen",
             "solved_geometric_median_truefrozen_ogl3.pt"),
    "gs": ("gaussian", "recon_remote/gs_froz/{s}/ckpt.pt",
           "solved_geometric_median_gs_froz_ogl3.pt"),
}


def cell_gt(kind, ckpt, gt_pts, gt_lab, n_classes):
    """Majority GT label per primitive, under each representation's own ownership query."""
    from oracle_labels import oracle_labels, oracle_labels_nearest
    if kind == "powerfoam":
        from build_true_facet_graph import load_points_radii
        cc, rr = load_points_radii(ckpt)
        return oracle_labels(np.asarray(cc, float), np.asarray(rr, float),
                             gt_pts, gt_lab, n_classes)[0]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)["splats"]
    return oracle_labels_nearest(ck["means"].float().numpy().astype(np.float64),
                                 gt_pts, gt_lab, n_classes)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--arms", nargs="*", default=["foam", "gs"])
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--min-cells", type=int, default=100,
                    help="a class needs this many GT cells before it is reported")
    ap.add_argument("--text-mode", choices=["raw","negproj","lowdin"], default="raw")
    ap.add_argument("--out", default="artifacts/confuser_diag_scannet.json")
    a = ap.parse_args()

    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                           load_scannet_pointcept_gt, remap_gt_labels)

    enable_determinism()
    names = list(OPENGAUSSIAN_CLASS_SETS[a.class_set])
    K = len(names)
    text = embed_class_names(names, "cuda")
    text = torch.nn.functional.normalize(text.float(), dim=-1)

    if a.text_mode != "raw":
        # `negproj`: project out the span of LERF's canonical negatives. The hypothesis is that a
        # shared "generic indoor thing" direction inflates every class-class cosine, so removing it
        # decorrelates the confuser pairs. Note this is NOT what LERF does with the negatives --
        # LERF uses them for a rejection test, which cannot break a tie between two real classes.
        # `lowdin`: symmetric (Lowdin) orthogonalisation, T <- (T T^T)^{-1/2} T. Makes prototypes
        # exactly orthogonal while staying as close as possible to the originals, so it removes ALL
        # pairwise redundancy rather than only the part lying in the negatives' span.
        if a.text_mode == "negproj":
            from semantic_reject import CANONICAL_NEGATIVES
            neg = torch.nn.functional.normalize(
                embed_class_names(CANONICAL_NEGATIVES, "cuda").float(), dim=-1)
            q, _ = torch.linalg.qr(neg.T)                      # orthonormal basis of the negatives
            text = torch.nn.functional.normalize(text - (text @ q) @ q.T, dim=-1)
        elif a.text_mode == "lowdin":
            g = text @ text.T
            ev, V = torch.linalg.eigh(g.double())
            inv_sqrt = (V * ev.clamp_min(1e-8).rsqrt()) @ V.T
            text = torch.nn.functional.normalize((inv_sqrt.float() @ text), dim=-1)
        print(f"[text] mode={a.text_mode}")
    tcos = (text @ text.T).cpu().numpy()
    np.fill_diagonal(tcos, -1.0)
    print(f"{K} classes, scoring is {K}-way over the full vocabulary\n")

    results = {}
    for arm in a.arms:
        kind, ckt, fn = ARMS[arm]
        # accumulate per class over scenes, in GLOBAL class ids so classes aggregate
        pred_of = {c: [] for c in range(1, K + 1)}
        margin_of = {c: [] for c in range(1, K + 1)}   # score_c - score_other, per class pair
        scores_of = {c: [] for c in range(1, K + 1)}
        n_scenes = 0
        for scene in a.scenes:
            ck = ckt.format(s=scene)
            fp = f"artifacts/scannet/{scene}/{fn}"
            exists = os.path.isdir(ck) if kind == "powerfoam" else os.path.exists(ck)
            if not (exists and os.path.exists(fp)):
                print(f"[miss] {arm}/{scene}")
                continue
            pts, raw, all_names = load_scannet_pointcept_gt(
                os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
            n2i = {n: i for i, n in enumerate(all_names)}
            gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])   # 1..K global ids
            cg = cell_gt(kind, ck, pts, gt_lab, K + 1)

            d = torch.load(fp, map_location="cpu", weights_only=True)
            f = d["primitive_features"].float()
            vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(len(f), bool)
            # a zero-norm feature has no direction; arg-max would silently hand it class 1
            nz = f.norm(dim=-1).numpy() > 1e-8
            live = vm & nz & (cg > 0)
            if live.sum() == 0:
                continue
            s = (torch.nn.functional.normalize(f, dim=-1).cuda() @ text.T).cpu().numpy()
            pr = s.argmax(1) + 1
            for c in range(1, K + 1):
                m = live & (cg == c)
                if not m.any():
                    continue
                pred_of[c].append(pr[m])
                scores_of[c].append(s[m])
            n_scenes += 1

        rows = []
        for c in range(1, K + 1):
            if not pred_of[c]:
                continue
            pr = np.concatenate(pred_of[c])
            sc = np.concatenate(scores_of[c])
            if len(pr) < a.min_cells:
                continue
            accK = float((pr == c).mean() * 100)
            wrong = pr[pr != c]
            if wrong.size == 0:
                conf, share = None, 0.0
            else:
                u, n = np.unique(wrong, return_counts=True)
                conf = int(u[n.argmax()]); share = float(n.max() / len(pr) * 100)
            if conf is None:
                continue
            # the 2-way contest: only these two columns, cells of the true class
            two = float((sc[:, c - 1] > sc[:, conf - 1]).mean() * 100)
            rows.append({
                "class": names[c - 1], "n_cells": int(len(pr)), "accK": accK,
                "confuser": names[conf - 1], "confuser_share": share,
                "acc_2way": two, "text_cos": float(tcos[c - 1, conf - 1]),
            })
        rows.sort(key=lambda r: r["accK"])
        results[arm] = rows

        print(f"=== {arm}  ({n_scenes} scenes) " + "=" * 40)
        print(f"{'class':>16} {'n':>7} {'accK':>7} {'2-way':>7} {'confuser':>16} "
              f"{'share':>7} {'txtcos':>7}  verdict")
        for r in rows:
            v = ("INFORMATION (<=50, unfixable by decision rules)" if r["acc_2way"] <= 55
                 else "COMPETITION (evidence present, hub wins)" if r["accK"] < 40
                 else "ok")
            print(f"{r['class']:>16} {r['n_cells']:>7,} {r['accK']:>7.1f} {r['acc_2way']:>7.1f} "
                  f"{r['confuser']:>16} {r['confuser_share']:>6.1f}% {r['text_cos']:>7.3f}  {v}")
        fail = [r for r in rows if r["accK"] < 40]
        if fail:
            below = [r for r in fail if r["acc_2way"] <= 50]
            print(f"\n  failing classes (accK<40): {len(fail)}   "
                  f"mean 2-way {np.mean([r['acc_2way'] for r in fail]):.1f}%   "
                  f"of which BELOW chance: {len(below)}")
            print(f"  mean text_cos to confuser, failing: "
                  f"{np.mean([r['text_cos'] for r in fail]):.3f}   "
                  f"passing: {np.mean([r['text_cos'] for r in rows if r['accK'] >= 40]):.3f}")
        print()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")

    if len(results) == 2 and all(results.values()):
        f = {r["class"]: r for r in results.get("foam", [])}
        g = {r["class"]: r for r in results.get("gs", [])}
        both = [c for c in f if c in g]
        print(f"\n=== shared classes ({len(both)}) " + "=" * 34)
        print(f"{'class':>16} {'foam accK':>10} {'gs accK':>9} {'foam 2way':>10} {'gs 2way':>9} "
              f"{'same confuser?':>15}")
        same = 0
        for c in sorted(both, key=lambda x: f[x]["accK"]):
            s = f[c]["confuser"] == g[c]["confuser"]
            same += s
            print(f"{c:>16} {f[c]['accK']:>10.1f} {g[c]['accK']:>9.1f} "
                  f"{f[c]['acc_2way']:>10.1f} {g[c]['acc_2way']:>9.1f} {str(s):>15}")
        print(f"\nsame top confuser on both representations: {same}/{len(both)} classes")
        print("If the confusers agree, the failure is in the TEXT/vocabulary side and is "
              "representation-independent.")


if __name__ == "__main__":
    main()
