"""VIEW QUORUM with exact cell attribution: the foam version of distractor filtering.

WHY THIS, AND WHY FOAM. arXiv 2608.26951 filters distractors by asking which content is inconsistent
with the other views, then verifying by removal. Its attribution step relies on a per-view
decomposition of the representation. In an overlapping Gaussian mixture, "which primitive produced
this pixel" is genuinely ambiguous -- unbounded support means many Gaussians contribute at the same
depth with soft weights. In a power diagram the cells are DISJOINT and BOUNDED, so a ray has an
unambiguous ordered cell sequence with exact segment weights `A[r,j]`, and the evidence a pixel
carries can be attributed to a specific cell rather than smeared over a blend.

WHAT MAKES IT WORTH DOING NOW. `run_view_split_diag.py` refuted my own prediction: solving the field
on two disjoint halves of the views flips the predicted class for 42% of the ERRORS. The evidence is
not view-consistent, so there IS something for a consistency method to recover -- contrary to what
the n_eff~35 figure suggested.

THE METHOD, and why it is not the feature solve. Today every view's mask embedding is averaged into
one 512-d vector per cell (geometric median over views), and only then classified. Averaging in CLIP
space is exactly where a hypernym wins: a view that clearly shows `kitchen cabinet` and one showing
generic `cabinet` blend into something nearer `cabinet`, and the specific evidence is destroyed
before the classifier ever sees it (measured: `kitchen cabinet` vs `cabinet` prototypes at cosine
0.829, two-way accuracy 30.9%, BELOW chance).

Voting inverts the order: classify FIRST, per view, then aggregate in LABEL space.

    votes[j, c] = sum_r A[r,j] * 1[ argmax_c' cos(mask_embed(r), t_c') == c ]

A view that clearly sees the specific class casts a full vote for it, and cannot be averaged away.
This is N3 ("view-quorum labeling") from the plan file, never implemented.

It also yields the distractor signal for free: `purity[j] = max_c votes[j,c] / sum_c votes[j,c]`.
A cell whose views disagree is exactly the "inconsistent content" the paper looks for, and because
the partition is exact we know precisely WHICH cell it is.

IMPLEMENTATION NOTE. Reuses the verified payload trick from `run_attribution_diag.py`: instead of a
512-d CLIP map, the per-pixel payload is a C-dim vote vector, so the existing accumulation operator
does the ray-cell attribution with no new traversal code. The class set is therefore baked in at
accumulation time, which is the one real cost of voting over averaging.

    D:/conda/envs/powerfoam/python.exe run_view_quorum_eval.py --stage accumulate --scene X
    python run_view_quorum_eval.py --stage analyze --scene X
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

RECON = os.path.join("D:" + os.sep, "Downloads", "spp_results", "full")
FEAT_ROOT = Path("D:" + os.sep) / "Downloads" / "spp_data_1600"


def log(m):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def vote_map(mask_feat, seg, txt, soft=0.0):
    """(H,W,C+1) payload: per-pixel vote over classes, plus a constant channel for normalisation.

    `soft=0` casts a hard one-hot vote for the argmax class -- the point of voting is that a
    confident view is not diluted by an ambiguous one. `soft>0` uses softmax(cos/soft) instead, kept
    as an arm because every hard correction in this project has lost to its partial version.
    """
    C = txt.shape[0]
    pad = torch.cat([torch.zeros(1, mask_feat.shape[1], device=mask_feat.device), mask_feat], 0)
    fm = torch.nn.functional.embedding(seg, pad).sum(0)
    fm = fm / (fm.norm(dim=-1, keepdim=True) + 1e-6)
    cos = fm @ txt.T                                   # (H,W,C)
    if soft > 0:
        v = torch.softmax(cos / soft, dim=-1)
    else:
        v = torch.zeros_like(cos).scatter_(-1, cos.argmax(-1, keepdim=True), 1.0)
    v = v * (seg > 0).any(0).unsqueeze(-1).float()     # unmasked pixels cast no vote
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def stage_accumulate(scene, n_cls=100, soft=0.0, device="cuda"):
    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import accumulate_feature_stats_for_views
    from evaluate_point_cloud_miou import embed_class_names
    from run_spp_eval import benchmark_map

    wp.init()
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args)
    dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ck}/model.pt")
    images_dir = Path(args.data_path) / args.scene / "images"
    image_names = sorted(p_.stem for p_ in images_dir.iterdir())
    top, _ = benchmark_map()
    names = top[:n_cls]
    txt = embed_class_names(names, device)
    folder = FEAT_ROOT / scene / "openclip_features_sam_l3"

    def loader(view_id):
        stem = image_names[view_id]
        fp, sp = folder / f"{stem}_f.npy", folder / f"{stem}_s.npy"
        if not fp.exists():
            return torch.zeros(1066, 1600, n_cls + 1, device=device)
        feat = torch.from_numpy(np.load(fp)).to(device).float()
        seg = torch.from_numpy(np.load(sp)).to(device).long() + 1
        return vote_map(feat, seg, txt, soft)

    st = accumulate_feature_stats_for_views(model, dh.cameras, list(range(len(dh.cameras))),
                                            loader, batch_size=1)
    num = st.numerator.float().cpu().numpy()
    tag = "hard" if soft == 0 else f"soft{soft:g}"
    out = f"artifacts/scannetpp/{scene}/votes_{tag}.npy"
    np.save(out, num)
    tot = num[:, :n_cls].sum(1)
    seen = tot > 0
    purity = np.zeros_like(tot)
    purity[seen] = num[seen, :n_cls].max(1) / tot[seen]
    log(f"  saved {out}: {seen.sum():,} cells voted on; median purity {np.median(purity[seen]):.3f}")


def stage_analyze(scene, out_json, n_cls=100, device="cuda"):
    import torch.nn.functional as F
    from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
    from feature_foam_lifting.operator import AccumulatedFeatureStats
    from point_cloud_query import assign_points_to_power_cells
    from build_true_facet_graph import load_points_radii
    from run_simplex_diffusion_eval import csr_to_edges, diffuse
    from run_derived_stack_eval import rank_encode
    from run_normlift_refine_eval import mode_vote_refine
    from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS, score_pred
    from run_spp_eval import benchmark_map, load_gt, coverage_filter

    art = f"artifacts/scannetpp/{scene}"
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    centers, radii = load_points_radii(ck)
    votes_all = np.load(f"{art}/votes_hard.npy")
    sv = torch.load(f"{art}/solved_geometric_median_nonfrozen_ogl3.pt",
                    map_location=device, weights_only=True)
    feats = sv["primitive_features"].to(device).float()
    vmn = sv["valid_mask"].cpu().numpy(); vm = torch.from_numpy(vmn).to(device)
    P = feats.shape[0]
    raw = torch.zeros_like(feats); raw[vm] = F.normalize(feats[vm], dim=-1)
    del feats, sv
    R = (AccumulatedFeatureStats.load(f"{art}/stats_nonfrozen_ogl3.pt")
         .reliability()["reliability"].to(device).float() * vm)
    pos = torch.from_numpy(centers).to(device).float()
    mu = F.normalize(raw[vm].mean(0, keepdim=True), dim=-1)
    cen = raw.clone(); cen[vm] = F.normalize(raw[vm] - LAM * mu, dim=-1)
    adj = torch.load(f"{art}/adjacency_true_facet.pt", map_location=device, weights_only=True)
    ad0, of0 = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
    Dm = int((of0[1:] - of0[:-1]).max()) + 1
    cen = mode_vote_refine(cen, R, pos, ad0, of0, chunk=max(256, 200_000 // max(Dm, 1)))
    src, dst, _ = csr_to_edges(ad0, of0, P, device)
    ke = vm[src] & vm[dst]; src, dst = src[ke], dst[ke]
    deg = torch.zeros(P, dtype=torch.long, device=device).index_add_(0, src, torch.ones_like(src))

    top, r2b = benchmark_map()
    gt_pts, lab0, _ = load_gt(scene, top, r2b)
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=vmn, k=64)
    owned = assigned >= 0
    keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
    lab = np.where(keepc, lab0, -1)
    res = {}
    for K in (100, 20):
        pres = sorted(set(np.unique(lab).tolist()) & set(range(K)))
        if not pres: continue
        nm = [top[:K][i] for i in pres]
        gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
        txt = embed_class_names(nm, device); C = len(nm)
        V = torch.from_numpy(votes_all[:, pres]).to(device).float()   # votes for the present classes
        tot = V.sum(1)
        purity = torch.zeros_like(tot); s = tot > 0
        purity[s] = V[s].max(1).values / tot[s]

        cv = cen[vm] @ txt.T
        rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
        base = torch.zeros(P, C, device=device); base[vm] = cv - 0.5 * rK[None, :]

        def finish(scores):
            p0 = rank_encode(scores, RANK_S, device); p0[~vm] = 0.0
            x = diffuse(p0, src, dst, deg, ALPHA, ITERS)
            return score_pred(x.argmax(-1).cpu().numpy(), assigned, owned,
                              gt_t, C, gt_pts.shape[0])[0]

        r = {"base_feature_solve": finish(base)}
        Vn = V / tot.clamp_min(1e-9)[:, None]
        r["quorum_only"] = finish(Vn)
        # blends: votes carry LABEL evidence the feature average destroys, but are sparse
        zb = (base - base.mean(1, keepdim=True)) / base.std(1, keepdim=True).clamp_min(1e-8)
        zv = (Vn - Vn.mean(1, keepdim=True)) / Vn.std(1, keepdim=True).clamp_min(1e-8)
        for w in (0.25, 0.5, 0.75):
            r[f"blend_{w:g}"] = finish((1 - w) * zb + w * zv)
        # DISTRACTOR FILTER: trust votes only where the views agree, else fall back to the feature
        for thr in (0.5, 0.7):
            m = (purity >= thr)[:, None].float()
            r[f"quorum_if_pure{thr:g}"] = finish(m * zv + (1 - m) * zb)
        res[f"top{K}"] = r
        if K == 100:
            res["purity_median"] = float(purity[s].median())
            res["frac_pure_0.5"] = float((purity[s] >= 0.5).float().mean())
        log(f"  top{K}: " + " ".join(f"{k}={v:.2f}" for k, v in r.items()))
        del txt, cv, base, V
        torch.cuda.empty_cache()
    json.dump(res, open(out_json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["accumulate", "analyze"], required=True)
    p.add_argument("--scene", default="f9f95681fd")
    p.add_argument("--soft", type=float, default=0.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.stage == "accumulate":
        stage_accumulate(a.scene, soft=a.soft)
    else:
        stage_analyze(a.scene, a.out or f"artifacts/scannetpp/quorum_{a.scene}.json")


if __name__ == "__main__":
    main()
