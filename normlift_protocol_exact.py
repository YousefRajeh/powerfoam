"""NormLift's ScanNet protocol EXACTLY as their eval script implements it.

Read from `D:\\Downloads\\my_eval_scannet2.py` (their evaluator), which differs from what this
project had been doing in two ways, both of which cost us points:

  1. NO ASSIGNMENT. It hard-requires `M == N_gt` and then indexes straight through --
     "第 i 个 GT point 的 Gaussian 就是 splats[i] / gauss_feats[i]. 不做额外 NN 查询."
     Frozen init gives one Gaussian per GT vertex, so correspondence is the IDENTITY. We had been
     applying Mahalanobis (which is Dr.Splat's rule, a different paper) on top of an already
     exact 1:1 mapping -- every mismatch it introduced was pure loss. Verified 1:1 on all 10
     scenes: gs_froz primitive counts equal the GT point counts exactly.

  2. OPACITY CULL, which we omitted entirely:
         low_opacity = sigmoid(gauss_opacities) < 0.1
         gt_tensor[low_opacity] = 0        # -> ignored, exactly OpenGaussian's eval_scannet.py:128
     Note this zeroes the GT LABEL, so those points leave the metric altogether rather than being
     counted as errors.

Both are their published protocol and are applied verbatim here. This is not a variant of the rule
and must not become one -- the whole point is comparability with 35.77 / 39.62 / 48.93.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       calculate_metrics, remap_gt_labels,
                                       load_scannet_pointcept_gt)
from run_cluster_classify_eval import SCENES as SN_SCENES
from run_normlift_refine_eval import mode_vote_refine
from normlift_replication import knn_csr_safe, PUBLISHED, CLASS_SETS

SN_GT = r"D:\Downloads\scannet_pointcept"
RECON = r"D:\Downloads\powerfoam\recon_remote"
OPACITY_THRESH = 0.1


def load_gs(scene, arm="gs_froz"):
    ck = torch.load(os.path.join(RECON, arm, scene, "ckpt.pt"),
                    map_location="cpu", weights_only=False)["splats"]
    return ck["means"].float(), ck["opacities"].float().reshape(-1)


def run(out_json="artifacts/scannet/normlift_exact.json", cull=True, identity=True):
    enable_determinism()
    device = "cuda"
    res = {}
    for scene in list(SN_SCENES):
        try:
            fp = f"artifacts/scannet/{scene}/solved_weighted_gs_froz_ogl3.pt"
            if not os.path.exists(fp):
                continue
            sv = torch.load(fp, map_location=device, weights_only=True)
            feats = sv["primitive_features"].to(device).float()
            vm = sv["valid_mask"].to(device)
            P = feats.shape[0]
            unit = torch.zeros_like(feats); unit[vm] = F.normalize(feats[vm], dim=-1)
            R = feats.norm(dim=-1) * vm                       # Eq. 8 numerator
            del feats
            means, opac = load_gs(scene)
            if means.shape[0] != P:
                print(f"[skip] {scene}: {P} vs {means.shape[0]}"); continue
            pos = means.to(device)
            adj, off = knn_csr_safe(pos, vm, K=30)
            Dm = int((off[1:] - off[:-1]).max()) + 1
            ref = mode_vote_refine(unit, R, pos, adj, off,
                                   chunk=max(256, 200_000 // max(Dm, 1)))
            gt, rawl, names_all = load_scannet_pointcept_gt(
                os.path.join(SN_GT, SN_SCENES[scene], scene), "segment20")
            if gt.shape[0] != P and identity:
                print(f"[skip] {scene}: GT {gt.shape[0]} != P {P}"); continue
            low = (torch.sigmoid(opac) < OPACITY_THRESH).numpy()
            n2i = {n: i for i, n in enumerate(names_all)}
            pres = set(np.unique(rawl).tolist())
            for cs in CLASS_SETS:
                kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in pres]
                tids = [i for i, _ in kept]; nm = [n for _, n in kept]
                g = remap_gt_labels(rawl, tids)
                if cull:
                    g = g.copy(); g[low] = 0          # their line: gt_tensor[low_opacity] = 0
                gt_t = torch.from_numpy(g).long()
                txt = embed_class_names(nm, device)
                for tag, u in (("raw", unit), ("normlift", ref)):
                    c = torch.zeros(P, len(nm), device=device); c[vm] = u[vm] @ txt.T
                    pred = torch.from_numpy(c.argmax(-1).cpu().numpy() + 1).long()  # identity map
                    _, miou, _, macc = calculate_metrics(gt_t, pred, len(nm) + 1)
                    res.setdefault(cs, {}).setdefault(tag, []).append(
                        (float(miou) * 100, float(macc) * 100))
                    del c
                del txt
            del unit, ref, R, pos, adj, off
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[err] {scene}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    print(f"\ngs_froz | identity assignment | cull={cull}   (their protocol)")
    for cs in CLASS_SETS:
        if cs not in res: continue
        a = np.array(res[cs]["normlift"]); b = np.array(res[cs]["raw"])
        pub = PUBLISHED[cs]
        print(f"  {cs:<16} raw {b[:,0].mean():6.2f} -> NL {a[:,0].mean():6.2f}/{a[:,1].mean():6.2f}"
              f" | published {pub[0]:.2f}/{pub[1]:.2f} | d {a[:,0].mean()-pub[0]:+.2f} (n={len(a)})")
    json.dump({k: {m: v for m, v in d.items()} for k, d in res.items()},
              open(out_json, "w"), indent=1)
    return res


if __name__ == "__main__":
    run()
