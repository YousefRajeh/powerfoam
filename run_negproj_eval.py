"""Does projecting out LERF's canonical negatives help under the STANDARD protocol?

The confuser diagnostic scored a 19-way arg-max over the full vocabulary, because a hub that never
appears in a scene is exactly what that diagnostic exists to expose. That is NOT the protocol our
headline numbers use, and the +4.4 macro-accuracy it reported is therefore not a claim about mIoU.

This scores the published protocol instead, matching run_radfoam_eval.py line for line:
  * classes = those PRESENT in that scene's GT, arg-max restricted to them;
  * bare arg-max via classify_primitives (no rejection);
  * opacity mask: cells with alpha < --alpha are unlabelled, alpha = 1 - exp(-sigma * 2r);
  * GT points assigned by exact power-cell ownership;
  * per-scene mIoU/mAcc averaged over the classes present in that scene, then over scenes.

The only thing that varies between arms is the text side, so any difference is attributable to the
projection and nothing else.

WHY THE PROJECTION MIGHT NOT SURVIVE THE PROTOCOL CHANGE. Restricting to present classes already
removes many of the hubs the projection was helping against -- if `picture` is not in the scene, it
cannot steal `wall`'s cells in the first place. So the honest prior is that the gain shrinks, and
the question is by how much.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")

ARMS = {
    "pf_truefrozen": ("output/scannet_{s}_truefrozen",
                      "solved_geometric_median_truefrozen_ogl3.pt"),
    "pf_nonfrozen": ("output/scannet_{s}_nonfrozen",
                     "solved_geometric_median_nonfrozen_ogl3.pt"),
}


def neg_basis(device="cuda"):
    from evaluate_point_cloud_miou import embed_class_names
    from semantic_reject import CANONICAL_NEGATIVES
    neg = Fn.normalize(embed_class_names(CANONICAL_NEGATIVES, device).float(), dim=-1)
    q, _ = torch.linalg.qr(neg.T)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--arm", default="pf_truefrozen")
    ap.add_argument("--class-sets", nargs="*",
                    default=["opengaussian19", "opengaussian15", "opengaussian10"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="artifacts/negproj_eval.json")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           embed_class_names, load_scannet_pointcept_gt,
                                           remap_gt_labels)
    from point_cloud_query import assign_points_to_power_cells

    enable_determinism()
    q = neg_basis()
    ckt, fn = ARMS[a.arm]
    out = {}

    for cs in a.class_sets:
        rows = []
        for scene in a.scenes:
            ck = ckt.format(s=scene)
            fp = f"artifacts/scannet/{scene}/{fn}"
            if not (os.path.isdir(ck) and os.path.exists(fp)):
                print(f"[miss] {scene}")
                continue
            pts, raw, all_names = load_scannet_pointcept_gt(
                os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
            n2i = {n: i for i, n in enumerate(all_names)}
            present = set(np.unique(raw).tolist())
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            if len(names) < 2:
                continue
            gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1

            cc, rr = load_points_radii(ck)
            centers = np.asarray(cc, dtype=np.float64)
            radii = np.asarray(rr, dtype=np.float64)
            sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
            sigma = Fn.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
            alpha = 1.0 - np.exp(-np.maximum(sigma, 0.0) * 2.0 * radii)

            d = torch.load(fp, map_location="cpu", weights_only=True)
            feats = d["primitive_features"].float().cuda()
            vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(len(centers), bool)
            owner = np.asarray(assign_points_to_power_cells(pts, centers, radii, valid=None, k=8))

            txt = Fn.normalize(embed_class_names(names, "cuda").float(), dim=-1)
            variants = {
                "raw": txt,
                "negproj": Fn.normalize(txt - (txt @ q) @ q.T, dim=-1),
            }
            rec = {"scene": scene, "n_classes": len(names)}
            for vname, t in variants.items():
                pcls = classify_primitives(feats, t).cpu().numpy() + 1
                pcls[~vm] = 0
                pcls[alpha < a.alpha] = 0
                pred = np.where(owner >= 0, pcls[np.clip(owner, 0, len(pcls) - 1)], 0)
                ious, accs = [], []
                for k in range(1, nc):
                    g = gt_lab == k
                    if not g.any():
                        continue
                    p = pred == k
                    inter = float((g & p).sum())
                    union = float((g | p).sum())
                    ious.append(inter / union if union else 0.0)
                    accs.append(inter / float(g.sum()))
                rec[f"{vname}_mIoU"] = float(np.mean(ious) * 100)
                rec[f"{vname}_mAcc"] = float(np.mean(accs) * 100)
            rec["d_mIoU"] = rec["negproj_mIoU"] - rec["raw_mIoU"]
            rows.append(rec)
            print(f"  {cs:16s} {scene:14s} {len(names):2d}cls  "
                  f"raw {rec['raw_mIoU']:6.2f}  negproj {rec['negproj_mIoU']:6.2f}  "
                  f"{rec['d_mIoU']:+6.2f}", flush=True)

        if rows:
            out[cs] = rows
            r = np.mean([x["raw_mIoU"] for x in rows])
            n = np.mean([x["negproj_mIoU"] for x in rows])
            ra = np.mean([x["raw_mAcc"] for x in rows])
            na = np.mean([x["negproj_mAcc"] for x in rows])
            win = sum(1 for x in rows if x["d_mIoU"] > 0)
            print(f"\n{cs}: {len(rows)} scenes   mIoU raw {r:.2f} -> negproj {n:.2f} "
                  f"({n - r:+.2f})   mAcc {ra:.2f} -> {na:.2f} ({na - ra:+.2f})   "
                  f"scenes improved {win}/{len(rows)}\n")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
