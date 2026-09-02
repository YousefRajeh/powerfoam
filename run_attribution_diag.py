"""Does a cell get the WRONG class because it listened to the wrong-granularity SAM mask?

THE HYPOTHESIS, and why it is the only one left. The decision rule is closed: 13 arms against
hubness ([[CSLS-paper-ideas-2026-08-31]]) and a measured macro-IoU headroom of +0.14 even for an
oracle that searches directly for the best per-cell labelling ([[run_macro_iou_gap]]). n_eff is ~35
views per cell, so the per-cell feature is not noisy -- it is stably wrong. The remaining suspect is
ATTRIBUTION: each cell averages the CLIP embedding of whichever SAM mask covers it in each view, and
SAM masks are wildly heterogeneous in scale (f9f95681fd/DSC08422: largest mask 23.9% of the image,
top-3 48.1%, median 0.38%). A cell on a shelf covered by the `wall` mask in most views receives a
confident, view-consistent, WRONG feature -- exactly the observed failure mode, whole coherent
regions taking a hub class.

THE MEASUREMENT. Per cell, the render-weighted mean AREA of the masks that fed it, against whether
that cell is classified correctly. If the hypothesis holds, cells fed by large masks are wrong far
more often. If not, attribution is the wrong lever and the direction dies for the price of one
streaming pass -- which is why this runs before anything is built.

HOW. Rather than reimplement ray-cell traversal, reuse `accumulate_feature_stats_for_views` with a
2-channel per-pixel payload [mask_area, 1] instead of the usual 512-d CLIP map. The ratio
numerator[:,0]/numerator[:,1] is the render-weighted mean area and is invariant to any per-ray
scaling the operator applies, since both channels scale together.

TWO STAGES, TWO INTERPRETERS. `powerfoam.scene` needs fpsample+warp (conda env `powerfoam`); the
frozen stack needs open_clip+sklearn (env `gs-view`). No interpreter has both, so:
    D:/conda/envs/powerfoam/python.exe run_attribution_diag.py --stage accumulate --scene X
    python run_attribution_diag.py --stage analyze --scene X
exchanging artifacts/scannetpp/X/mask_area.npy.
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


def mask_area_map(seg, n_masks):
    """(H,W,2) payload: channel 0 = area fraction of the mask at that pixel, channel 1 = 1.

    Unmasked pixels (-1) get area 0 but still carry channel 1, so they dilute the mean towards 0
    exactly as they dilute the CLIP feature -- the same treatment the real lifting gives them.
    """
    s = seg.reshape(seg.shape[-2], seg.shape[-1]).long()
    counts = torch.bincount((s + 1).flatten(), minlength=n_masks + 1).float()
    area = counts / float(s.numel())
    area[0] = 0.0                                    # id -1 -> slot 0 -> unmasked
    amap = area[(s + 1)]
    return torch.stack([amap, torch.ones_like(amap)], dim=-1)


# ------------------------------------------------------------------ stage 1: accumulate (powerfoam)

def stage_accumulate(scene, device="cuda"):
    import configargparse
    import warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import accumulate_feature_stats_for_views

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
    log(f"  {scene}: {len(dh.cameras)} cameras, {len(image_names)} images, "
        f"{model.points.shape[0]:,} primitives")

    folder = FEAT_ROOT / scene / "openclip_features_sam_l3"

    def loader(view_id):
        sp = folder / f"{image_names[view_id]}_s.npy"
        if not sp.exists():
            return torch.zeros(1066, 1600, 2, device=device)
        seg = torch.from_numpy(np.load(sp)).to(device)
        return mask_area_map(seg, max(int(seg.max().item()) + 1, 1)).to(device)

    stats = accumulate_feature_stats_for_views(
        model, dh.cameras, list(range(len(dh.cameras))), loader, batch_size=1)
    num = stats.numerator.float().cpu().numpy()
    out = f"artifacts/scannetpp/{scene}/mask_area.npy"
    np.save(out, num)
    seen = num[:, 1] > 0
    ma = num[seen, 0] / np.maximum(num[seen, 1], 1e-12)
    log(f"  saved {out}: {seen.sum():,}/{len(seen):,} cells covered, "
        f"median mean-mask-area {np.median(ma):.4f}")


# --------------------------------------------------------------------- stage 2: analyze (gs-view)

def stage_analyze(scene, out_json, device="cuda"):
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from evaluate_point_cloud_miou import embed_class_names, remap_gt_labels
    from feature_foam_lifting.operator import AccumulatedFeatureStats
    from point_cloud_query import assign_points_to_power_cells
    from build_true_facet_graph import load_points_radii
    from run_simplex_diffusion_eval import csr_to_edges, diffuse
    from run_derived_stack_eval import rank_encode
    from run_normlift_refine_eval import mode_vote_refine
    from run_overnight import LAM, CSLS_K, RANK_S, ALPHA, ITERS
    from run_macro_iou_gap import cell_histograms
    from run_spp_eval import benchmark_map, load_gt, coverage_filter

    num = np.load(f"artifacts/scannetpp/{scene}/mask_area.npy")
    seen = num[:, 1] > 0
    mean_area = num[:, 0] / np.maximum(num[:, 1], 1e-12)

    art = f"artifacts/scannetpp/{scene}"
    ck = os.path.join(RECON, f"spp_pf_unfroz_{scene}")
    centers, radii = load_points_radii(ck)
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
    keepc, _, _ = coverage_filter(gt_pts, assigned, centers, vmn, 20.0)
    lab = np.where(keepc, lab0, -1)
    pres = sorted(set(np.unique(lab).tolist()) & set(range(100)))
    gt_t = torch.from_numpy(remap_gt_labels(lab, pres)).long()
    C = len(pres)
    H, _ = cell_histograms(assigned, gt_t, len(centers), C)

    txt = embed_class_names([top[:100][i] for i in pres], device)
    cv = cen[vm] @ txt.T
    rK = cv.topk(min(CSLS_K, cv.shape[0]), dim=0).values.mean(0)
    full = torch.zeros(P, C, device=device); full[vm] = cv - 0.5 * rK[None, :]
    p0 = rank_encode(full, RANK_S, device); p0[~vm] = 0.0
    pred = diffuse(p0, src, dst, deg, ALPHA, ITERS).argmax(-1).cpu().numpy()

    m = (H.sum(1) > 0) & seen & vmn
    correct = pred[m] == H.argmax(1)[m]
    area = mean_area[m]
    log(f"  evaluable cells: {m.sum():,}  cell-level accuracy {correct.mean()*100:.2f}%")

    order = np.argsort(area)
    rows = [{"decile": i + 1, "mean_area": float(area[d].mean()), "n": int(len(d)),
             "acc": float(correct[d].mean() * 100)}
            for i, d in enumerate(np.array_split(order, 10))]
    print(f"\n{'decile':>7}{'mean mask area':>16}{'cells':>10}{'accuracy':>10}")
    for r in rows:
        print(f"{r['decile']:>7}{r['mean_area']:>16.4f}{r['n']:>10,}{r['acc']:>10.2f}")
    np.savez(f"artifacts/scannetpp/{scene}/attribution_arrays.npz",
             area=area, correct=correct)
    auc = float(roc_auc_score((~correct).astype(int), area))
    corr = float(np.corrcoef(area, (~correct).astype(float))[0, 1])
    print(f"\n  AUC(mask area -> error): {auc:.4f}   point-biserial r: {corr:+.4f}")
    print(f"  smallest-area decile {rows[0]['acc']:.2f}% vs largest {rows[-1]['acc']:.2f}%  "
          f"(spread {rows[0]['acc']-rows[-1]['acc']:+.2f})")
    print("\n  GATE: AUC > 0.65 with a monotone decline supports attribution as the lever;\n"
          "  AUC ~0.5 kills the direction.")
    json.dump({"scene": scene, "auc": auc, "corr": corr, "deciles": rows},
              open(out_json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["accumulate", "analyze"], required=True)
    p.add_argument("--scene", default="f9f95681fd")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.stage == "accumulate":
        stage_accumulate(a.scene)
    else:
        stage_analyze(a.scene, a.out or f"artifacts/scannetpp/attribution_{a.scene}.json")


if __name__ == "__main__":
    main()
