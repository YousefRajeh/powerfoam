"""Splat Feature Solver's Post-Lifting Aggregation detector, run on OUR foam scenes.

WHY REIMPLEMENT THEIR DETECTOR ON FOAM RATHER THAN RUN distill.py ON 3DGS. distill.py's PLA path
(label_projection -> mask_level_refinement, distill.py:336-339) operates on Gaussian splats, so its
trust scores live on a DIFFERENT observation set than the (foam cell, view) pairs our detector AUC
was measured over. Two AUCs computed on different observations answer different questions and cannot
be compared. Running their ALGORITHM on our representation holds the scenes, the observations and the
contamination ground truth fixed, so the only thing that varies is the detector -- which is the
comparison that decides anything.

THEIR ALGORITHM, followed faithfully (paper Appendix + distill.py + gaussian_splatting/analysis.py):
  1. cluster the lifted features -- PCA to 50 dims then HDBSCAN, min_cluster_size=500, their defaults
  2. project cluster labels into every training frustum (their `label_projection`: one-hot labels
     pushed through the renderer; the arg-max of that is the dominant cluster per pixel, which is
     what we compute directly)
  3. for each input SAM mask, take the dominant projected cluster inside it and form that cluster's
     pseudo-mask, then trust(mask) = IoU(mask, pseudo-mask)
  4. keep observations whose trust exceeds a threshold (0.75 in the paper)

OUTPUT. trust per (cell, view), saved small. The AUC is computed back on the workstation by joining
against the contamination labels already measured there -- so no ScanNet GT is needed on this host.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--variant", default="truefrozen")
    ap.add_argument("--solved", default="solved_geometric_median_truefrozen_ogl3.pt")
    ap.add_argument("--features",
                    default="data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-components", type=int, default=50)      # their default
    ap.add_argument("--min-cluster-size", type=int, default=500)  # their default
    ap.add_argument("--out", default="artifacts/pla_trust_{scene}.npz")
    a = ap.parse_args()

    import configargparse
    import warp as wp

    from configs import Params, add_group
    from data_loader import DataHandler
    from determinism import enable_determinism
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene

    enable_determinism()
    ck = f"output/scannet_{a.scene}_{a.variant}"
    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]

    # ---- step 1: cluster the lifted features, their way -----------------------------------------
    d = torch.load(f"artifacts/scannet/{a.scene}/{a.solved}", map_location="cpu",
                   weights_only=True)
    F = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1).numpy()
    vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(P, bool)
    from cuml.cluster import HDBSCAN
    from cuml.decomposition import PCA
    X = F[vm].astype(np.float32)
    red = PCA(n_components=min(a.n_components, X.shape[1])).fit_transform(X)
    lab_v = HDBSCAN(min_cluster_size=a.min_cluster_size).fit_predict(red)
    lab_v = np.asarray(lab_v).astype(np.int32)
    clus = np.full(P, -1, np.int32)
    clus[vm] = lab_v
    nclus = int(clus.max()) + 1
    print(f"HDBSCAN: {nclus} clusters, noise {(clus < 0).mean():.1%} of {P:,} cells", flush=True)
    if nclus <= 0:
        print("[abort] no clusters found"); return

    from accumulate_hard_mask import load_masks

    stems = sorted(p.stem for p in (Path(cargs.data_path) / cargs.scene / "images").iterdir())
    feat_dir = a.features.format(scene=a.scene)
    CID, VID, TRUST = [], [], []
    for vi, cam in enumerate(dh.cameras):
        H, W = int(cam.height), int(cam.width)
        _, seg = load_masks(feat_dir, stems[vi], a.level, H, W)
        seg = seg.reshape(-1).numpy()
        M = int(seg.max()) + 1
        if M <= 0:
            continue
        op = export_operator_for_views(model, [cam], [vi])
        rows = op.row_indices.cpu().numpy()
        cols = op.col_indices.cpu().numpy()
        vals = op.values.cpu().numpy().astype(np.float64)

        # ---- step 2: dominant projected CLUSTER per pixel (arg-max of projected one-hot) --------
        npx = H * W
        pix_lab = np.full(npx, -1, np.int32)
        best = np.zeros(npx)
        ok = clus[cols] >= 0
        r_, c_, v_ = rows[ok], cols[ok], vals[ok]
        order = np.lexsort((-v_, r_))
        r_s, c_s, v_s = r_[order], c_[order], v_[order]
        first = np.ones(len(r_s), bool)
        first[1:] = r_s[1:] != r_s[:-1]
        pix_lab[r_s[first]] = clus[c_s[first]]
        del r_, c_, v_, r_s, c_s, v_s

        # ---- step 3: per mask, dominant cluster and IoU against its pseudo-mask -----------------
        good = (seg >= 0) & (pix_lab >= 0)
        h = np.zeros((M, nclus), np.int64)
        np.add.at(h, (seg[good], pix_lab[good]), 1)
        dom = h.argmax(1)
        inter = h.max(1).astype(np.float64)
        mask_area = np.bincount(seg[seg >= 0], minlength=M).astype(np.float64)
        clus_area = np.bincount(pix_lab[pix_lab >= 0], minlength=nclus).astype(np.float64)
        union = mask_area + clus_area[dom] - inter
        trust = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        trust[h.sum(1) == 0] = 0.0

        # ---- step 4: attach each (cell, view) observation the trust of the mask it reads --------
        mm = seg[rows]
        keep = mm >= 0
        cc, vv, mmk = cols[keep], vals[keep], mm[keep]
        if len(cc) == 0:
            continue
        key = cc * M + mmk
        hh = np.bincount(key, weights=vv, minlength=P * M).reshape(P, M)
        Wv = hh.sum(1)
        best_m = hh.argmax(1)
        pres = Wv > 1e-9
        CID.append(np.where(pres)[0])
        VID.append(np.full(int(pres.sum()), vi))
        TRUST.append(trust[best_m[pres]])
        if vi % 10 == 0:
            print(f"  view {vi}: {M} masks, mean trust {trust.mean():.3f}, "
                  f"{int(pres.sum()):,} observations", flush=True)

    CID = np.concatenate(CID); VID = np.concatenate(VID); TRUST = np.concatenate(TRUST)
    print(f"\n{len(TRUST):,} (cell,view) observations")
    print(f"trust: mean {TRUST.mean():.4f}  median {np.median(TRUST):.4f}  "
          f"frac > 0.75 (their keep rule) = {(TRUST > 0.75).mean():.4f}")
    out = a.out.format(scene=a.scene)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, cell=CID.astype(np.int32), view=VID.astype(np.int32),
                        trust=TRUST.astype(np.float32), n_clusters=nclus)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
