"""PLA's detector on 3DGS -- the representation it was designed for.

THE DEFENCE BEING TESTED. On foam, Post-Lifting Aggregation's trust score came out anti-correlated
with contamination at every min_cluster_size from 10 to 1000, and its keep rule raised eps in all
seven settings. The available defence is that PLA is tuned for million-splat Gaussian scenes, where
HDBSCAN behaves differently -- on 51,610 foam cells it left 60-78% of cells as noise. This runs the
identical sweep on the 3DGS arm of the same scene, with the same cuML HDBSCAN, the same PCA-to-50,
the same trust definition and the same threshold, so the only thing that changes is the
representation.

Everything is per (gaussian, view), mirroring the foam script's per (cell, view). The contamination
ground truth needs ScanNet labels, which live on the workstation, so this host only produces trust
scores and the join happens there.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsplat_baseline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0062_00")
    ap.add_argument("--gs-arm", default="gs_froz")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--solved", default="solved_geometric_median_gs_froz_ogl3.pt")
    ap.add_argument("--features",
                    default="data/scannet/{scene}_colmap/openclip_features_sam_blackboth")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--min-cluster-size", type=int, default=500)
    ap.add_argument("--max-hits", type=int, default=64)
    ap.add_argument("--out", default="artifacts/pla_gs_trust_mcs{mcs}.npz")
    a = ap.parse_args()

    from accumulate_hard_mask import load_masks
    from determinism import enable_determinism
    from export_gsplat_operator import export_view_operator

    enable_determinism()
    ck = a.ckpt or f"recon_remote/{a.gs_arm}/{a.scene}/ckpt.pt"
    sp = torch.load(ck, map_location="cuda", weights_only=False)
    sp = sp["splats"] if "splats" in sp else sp
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]).reshape(-1)
    colors = sp["sh0"].reshape(len(means), 3)
    P = means.shape[0]
    print(f"{P:,} gaussians from {Path(ck).name}", flush=True)

    d = torch.load(f"artifacts/scannet/{a.scene}/{a.solved}", map_location="cpu",
                   weights_only=True)
    F = torch.nn.functional.normalize(d["primitive_features"].float(), dim=-1).numpy()
    vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(P, bool)
    assert len(F) == P, (len(F), P)

    try:
        from cuml.cluster import HDBSCAN
        from cuml.decomposition import PCA
        impl = "cuml"
    except Exception:
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import PCA
        impl = "sklearn"
    X = F[vm].astype(np.float32)
    red = PCA(n_components=min(a.n_components, X.shape[1])).fit_transform(X)
    lab_v = np.asarray(HDBSCAN(min_cluster_size=a.min_cluster_size).fit_predict(red)).astype(np.int32)
    clus = np.full(P, -1, np.int32)
    clus[vm] = lab_v
    nclus = int(clus.max()) + 1
    print(f"HDBSCAN[{impl}]: {nclus} clusters, noise {(clus < 0).mean():.1%} of {P:,} gaussians", flush=True)
    if nclus <= 0:
        print("[abort] no clusters"); return

    # cameras: reuse the shared bridge output written by the foam side so both representations
    # measure the identical rays
    cams = np.load(f"artifacts/participation/{a.scene}_cams_all.npz") \
        if os.path.exists(f"artifacts/participation/{a.scene}_cams_all.npz") else None
    if cams is None:
        raise SystemExit("need artifacts/participation/<scene>_cams.npz (from measure_participation)")
    K = torch.as_tensor(cams["K"], dtype=torch.float32, device="cuda")
    vms = torch.as_tensor(cams["viewmats"], dtype=torch.float32, device="cuda")
    view_ids = cams["view_ids"]
    W_, H_ = (int(x) for x in cams["wh"])
    stems = sorted(p.stem for p in Path(a.features.format(scene=a.scene)).parent.joinpath("images").iterdir())

    feat_dir = a.features.format(scene=a.scene)
    CID, VID, TRUST = [], [], []
    for k in range(vms.shape[0]):
        vi = int(view_ids[k])
        _, seg = load_masks(feat_dir, stems[vi], a.level, H_, W_)
        seg = seg.reshape(-1).numpy()
        M = int(seg.max()) + 1
        if M <= 0:
            continue
        r, c, v, _, _ = export_view_operator(means, quats, scales, opac, colors,
                                             vms[k], K, W_, H_, max_hits_per_pixel=a.max_hits)
        rows = r.cpu().numpy(); cols = c.cpu().numpy(); vals = v.cpu().numpy().astype(np.float64)

        npx = H_ * W_
        pix_lab = np.full(npx, -1, np.int32)
        ok = clus[cols] >= 0
        r_, c_, v_ = rows[ok], cols[ok], vals[ok]
        order = np.lexsort((-v_, r_))
        r_s, c_s = r_[order], c_[order]
        first = np.ones(len(r_s), bool)
        first[1:] = r_s[1:] != r_s[:-1]
        pix_lab[r_s[first]] = clus[c_s[first]]

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

        mm = seg[rows]
        keep = mm >= 0
        cc, vv, mmk = cols[keep], vals[keep], mm[keep]
        if len(cc) == 0:
            continue
        key = cc.astype(np.int64) * M + mmk
        hh = np.bincount(key, weights=vv, minlength=P * M).reshape(P, M)
        Wv = hh.sum(1); best_m = hh.argmax(1)
        pres = Wv > 1e-9
        CID.append(np.where(pres)[0]); VID.append(np.full(int(pres.sum()), vi))
        TRUST.append(trust[best_m[pres]])
        print(f"  view {vi}: {M} masks, mean trust {trust.mean():.3f}, "
              f"{int(pres.sum()):,} obs", flush=True)

    CID = np.concatenate(CID); VID = np.concatenate(VID); TRUST = np.concatenate(TRUST)
    print(f"\n{len(TRUST):,} (gaussian,view) observations")
    print(f"trust: mean {TRUST.mean():.4f}  median {np.median(TRUST):.4f}  "
          f"frac > 0.75 = {(TRUST > 0.75).mean():.4f}")
    out = a.out.format(mcs=a.min_cluster_size)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, cell=CID.astype(np.int32), view=VID.astype(np.int32),
                        trust=TRUST.astype(np.float32), n_clusters=nclus)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
