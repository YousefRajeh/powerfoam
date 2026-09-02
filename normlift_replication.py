"""NormLift applied to EVERY reconstruction, validated against its published ScanNet table.

WHY EVERY ARM. NormLift is representation-agnostic: weighted lift (Eq. 5) + reliability (Eq. 8) +
reliability-guided KNN mode-voting (Eqs. 9-10) act on a primitive set and never assume Gaussians.
That is precisely what makes it a fair baseline to run on PowerFoam, RadFoam and 3DGS alike. Their
own numbers come from 3DGS with each Gaussian queried independently, so `gs_froz` is the row that
should reproduce 35.77 / 39.62 / 48.93.

WHAT THE PDF ACTUALLY SAYS (read, not inferred):
    Eq. (5)   u*_j = f_j / ||f_j||, with f_j weight-normalised by sum_i A_ij  -> the WEIGHTED
              solver. This project defaults to geometric_median, so every earlier "NormLift
              refinement" run here sat on the wrong base features.
    Eq. (8)   R(j) = ||f_j|| * Neff(j) / (Neff(j) + beta),  beta = 1.
    Eq. (9-10) mode-vote over Euclidean KNN; "K varies by less than 0.7 points across [12,60],
              already flat above K~30"; (sigma_d, tau, gamma, Delta) fixed across all experiments.

BOTH OPTIONS ARE RUN WHEREVER THEY EXIST, rather than picking one:
  * solver: `weighted` (theirs) AND `geometric_median` (ours). RadFoam has only gm, noted in the key.
  * graph : `knn30` (their Euclidean KNN) AND `delaunay` (each arm's exact dual), so the graph
            substitution is measured instead of assumed.

Assignments come from the cached ablation DB, so GT->primitive correspondence is each arm's own --
mahalanobis for Gaussians (Dr.Splat's rule, already implemented there) and power-cell for foam --
and is identical to every other experiment in this project.
"""
import json
import os
import sqlite3
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (
    OPENGAUSSIAN_CLASS_SETS, embed_class_names, calculate_metrics,
    remap_gt_labels, load_scannet_pointcept_gt,
)
from feature_foam_lifting.operator import AccumulatedFeatureStats
from build_true_facet_graph import load_points_radii
from run_cluster_classify_eval import SCENES as SN_SCENES
from run_normlift_refine_eval import mode_vote_refine, build_knn_csr

SN_DB = r"D:\Downloads\powerfoam\artifacts\ablation.sqlite"
SN_GT = r"D:\Downloads\scannet_pointcept"
PUBLISHED = {"opengaussian19": (35.77, 54.02), "opengaussian15": (39.62, 59.26),
             "opengaussian10": (48.93, 68.83)}
CLASS_SETS = ["opengaussian19", "opengaussian15", "opengaussian10"]
# recon -> the token used in this repo's solved-artifact filenames
NAME = {"pf_nonfroz": "nonfrozen", "pf_tfroz": "truefrozen",
        "gs_froz": "gs_froz", "gs_unfroz": "gs_unfroz",
        "rf_unfroz": "rf_unfroz", "rf_froz": "rf_froz"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_positions(recon, ckpt_path):
    """Primitive centres. Each representation persists them differently."""
    if recon.startswith("pf_"):
        c, _ = load_points_radii(os.path.dirname(ckpt_path))
        return np.asarray(c)
    if recon.startswith("rf"):
        from radfoam_adapter import load_radfoam_foam
        c, _ = load_radfoam_foam(ckpt_path)
        return np.asarray(c)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def search(o, depth=0):
        """gsplat writes {'step': int, 'splats': OrderedDict{'means': ...}}, so a flat scan of
        top-level values misses the tensor entirely -- recurse."""
        if depth > 4:
            return None
        if torch.is_tensor(o):
            return o if (o.ndim == 2 and o.shape[1] == 3) else None
        if isinstance(o, dict):
            for k in ("means", "xyz", "_xyz", "means3D"):
                v = o.get(k)
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == 3:
                    return v
            for v in o.values():
                r = search(v, depth + 1)
                if r is not None:
                    return r
        elif isinstance(o, (list, tuple)):
            for v in o:
                r = search(v, depth + 1)
                if r is not None:
                    return r
        return None

    t = search(sd)
    if t is None:
        raise ValueError(f"no centres found in {ckpt_path}")
    return np.asarray(t.float().cpu().numpy())


def knn_csr_safe(positions, valid_mask_t, K=30, max_elems=2.0e8):
    """K nearest neighbours without the 97 GB allocation.

    `build_knn_csr` blocks its cdist at 8192 ROWS, but the tensor is block x P_valid, so at
    gs_unfroz's 2.97M primitives one block is ~97 GB and the run dies -- then CUBLAS faults and
    poisons the context for everything after. For large scenes an exact CPU KD-tree is both
    memory-safe and faster than shrinking the block to ~66 rows; results are identical because
    both return exact Euclidean K-NN.
    """
    P = positions.shape[0]
    n_valid = int(valid_mask_t.sum())
    if P * n_valid <= max_elems:
        return build_knn_csr(positions, valid_mask_t, K=K)
    from scipy.spatial import cKDTree
    device = positions.device
    vi = torch.where(valid_mask_t)[0]
    vpos = positions[vi].detach().cpu().numpy()
    tree = cKDTree(vpos)
    _, idx = tree.query(positions.detach().cpu().numpy(), k=min(K + 1, n_valid), workers=-1)
    idx = np.atleast_2d(idx)[:, 1:K + 1]                 # drop self
    neigh = vi.detach().cpu().numpy()[idx]
    adjacent = torch.from_numpy(np.ascontiguousarray(neigh.reshape(-1))).to(device).long()
    offsets = torch.arange(0, P * neigh.shape[1] + 1, neigh.shape[1],
                           device=device, dtype=torch.long)
    return adjacent, offsets


def feature_path(scene, recon, solver):
    tok = NAME[recon]
    for cand in (f"artifacts/scannet/{scene}/solved_{solver}_{tok}_ogl3.pt",
                 f"artifacts/scannet/{scene}/solved_gm_{tok}_ogl3.pt"):
        if os.path.exists(cand):
            # only accept the gm-named legacy file when gm was actually requested
            if "solved_gm_" in cand and solver != "geometric_median":
                continue
            return cand
    return None


def score(cls_np, asg, own, gt_t, n_cls, n_gt):
    pred = np.zeros(n_gt, dtype=np.int64)
    pred[own] = cls_np[asg[own]] + 1
    _, miou, _, macc = calculate_metrics(gt_t, torch.from_numpy(pred).long(), n_cls + 1)
    return float(miou) * 100, float(macc) * 100


def run(out_json, device="cuda",
        arms=("gs_froz", "gs_unfroz", "pf_nonfroz", "pf_tfroz", "rf_unfroz", "rf_froz"),
        solvers=("weighted", "geometric_median"), graphs=("knn30", "delaunay")):
    con = sqlite3.connect(SN_DB, timeout=60.0)
    res = {}
    for recon in arms:
        for solver in solvers:
            for graph in graphs:
                key = f"{recon}|{solver}|{graph}"
                for scene in list(SN_SCENES):
                    try:
                        rrow = con.execute("select ckpt_path from reconstructions "
                                           "where scene=? and recon=?", (scene, recon)).fetchone()
                        arow = con.execute("select path from assignments "
                                           "where scene=? and recon=?", (scene, recon)).fetchone()
                        fp = feature_path(scene, recon, solver)
                        if not (rrow and arow and fp):
                            continue
                        sv = torch.load(fp, map_location=device, weights_only=True)
                        feats = sv["primitive_features"].to(device).float()
                        vmn = sv["valid_mask"].cpu().numpy()
                        vm = torch.from_numpy(vmn).to(device)
                        P = feats.shape[0]
                        unit = torch.zeros_like(feats)
                        unit[vm] = F.normalize(feats[vm], dim=-1)
                        # Eq. 8. Prefer the accumulator's own reliability (it already applies the
                        # Neff shrinkage with beta=1); fall back to ||f|| when stats are absent for
                        # this arm, which is the same quantity minus the shrinkage term.
                        R = feats.norm(dim=-1) * vm
                        sp = f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt"
                        if recon == "pf_nonfroz" and os.path.exists(sp):
                            rr = AccumulatedFeatureStats.load(sp).reliability()["reliability"]
                            if rr.shape[0] == P:
                                R = rr.to(device).float() * vm
                        del feats
                        pos = torch.from_numpy(load_positions(recon, rrow[0])).to(device).float()
                        if pos.shape[0] != P:
                            del unit, R, pos; torch.cuda.empty_cache(); continue
                        if graph == "knn30":
                            adj, off = knn_csr_safe(pos, vm, K=30)
                        else:
                            grow = con.execute("select path from adjacency where scene=? and "
                                               "recon=? and complex='delaunay'",
                                               (scene, recon)).fetchone()
                            if not grow:
                                del unit, R, pos; torch.cuda.empty_cache(); continue
                            g = torch.load(grow[0], map_location=device, weights_only=True)
                            adj = g["adjacent"].to(device).long()
                            off = g["offsets"].to(device).long()
                        Dm = int((off[1:] - off[:-1]).max()) + 1
                        ref = mode_vote_refine(unit, R, pos, adj, off,
                                               chunk=max(256, 200_000 // max(Dm, 1)))
                        gt, rawl, names_all = load_scannet_pointcept_gt(
                            os.path.join(SN_GT, SN_SCENES[scene], scene), "segment20")
                        asg = np.load(arow[0])
                        own = asg >= 0
                        n2i = {n: i for i, n in enumerate(names_all)}
                        pres = set(np.unique(rawl).tolist())
                        for cs in CLASS_SETS:
                            kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs]
                                    if n2i[n] in pres]
                            tids = [i for i, _ in kept]; nm = [n for _, n in kept]
                            gt_t = torch.from_numpy(remap_gt_labels(rawl, tids)).long()
                            txt = embed_class_names(nm, device)
                            c0 = torch.zeros(P, len(nm), device=device); c0[vm] = unit[vm] @ txt.T
                            c1 = torch.zeros(P, len(nm), device=device); c1[vm] = ref[vm] @ txt.T
                            m0 = score(c0.argmax(-1).cpu().numpy(), asg, own, gt_t,
                                       len(nm), gt.shape[0])
                            m1 = score(c1.argmax(-1).cpu().numpy(), asg, own, gt_t,
                                       len(nm), gt.shape[0])
                            d = res.setdefault(key, {}).setdefault(cs, {})
                            d.setdefault("raw", []).append(m0)
                            d.setdefault("normlift", []).append(m1)
                            del c0, c1, txt
                        del unit, ref, R, pos, adj, off
                        torch.cuda.empty_cache()
                    except Exception as e:
                        log(f"  [err] {key} {scene}: {type(e).__name__}: {e}")
                        torch.cuda.empty_cache()
                if key in res:
                    for cs in CLASS_SETS:
                        if cs not in res[key]:
                            continue
                        a = np.array(res[key][cs]["normlift"])
                        b = np.array(res[key][cs]["raw"])
                        pub = PUBLISHED[cs][0]
                        log(f"  {key:<36}{cs:<15} raw {b[:, 0].mean():6.2f} -> NL "
                            f"{a[:, 0].mean():6.2f} | pub {pub:5.2f} | "
                            f"d {a[:, 0].mean() - pub:+6.2f} (n={len(a)})")
                    with open(out_json, "w") as fh:
                        json.dump(res, fh, indent=1)
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    return res
