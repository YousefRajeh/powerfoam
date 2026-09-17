"""Evidence-weighted shrinkage readout: the continuous form of abstention.

THE DEFECT IT TARGETS (raised by the user). The operator already encodes opacity -- `A_ij = alpha_j
T_j`, and the renderer stops a ray at `transmittance_floor`. But the closed form then divides it out:

    Y_j = AtS_j / D_j ,     D_j = sum_i A_ij

so a primitive whose TOTAL accumulated weight is 1e-3 emerges with exactly the same unit-sum class
distribution as one with weight 1e3, and `argmax` commits with identical confidence. Every live
primitive votes with equal authority no matter how faintly it was ever seen. That is why the
contaminated primitives are the low-`D` ones (A36), why abstaining on `D_j` bought +0.33, and why
OpenGaussian needs an `alpha < 0.1` GT mask at all -- a harsher version of the same instinct.

THE FIX. Shrink each primitive toward what its neighbours say, in proportion to how little evidence
it has of its own:

    m_j = ( sum_{k in adj(j)} AtS_k ) / ( sum_{k in adj(j)} D_k )        neighbour consensus
    Y_j = ( AtS_j + kappa * m_j ) / ( D_j + kappa )

`kappa` is measured in the same units as `D`, so it reads as "how much evidence is needed before a
primitive is believed over its neighbourhood". Two properties hold by construction and are asserted
at runtime rather than assumed:

  * kappa = 0 reproduces Eq. 6 EXACTLY -- so the sweep measures the idea, not an implementation.
  * `m_j` and `Y0_j` are both in the probability simplex, and the update is a convex combination of
    them with weights D_j/(D_j+kappa) and kappa/(D_j+kappa), so the result is in the simplex too.
    No projection is needed, unlike the least-squares route.

Unlike every solver tried so far this minimises NO residual -- which by A36's pattern is a point in
its favour, since the one method that does not chase a residual (geometric median) is the one method
that has never lost. `--rounds > 1` iterates it, which is label propagation over the facet graph.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, calculate_metrics)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from diagnose_holes import SCENES, GT_ROOT
from oracle_projected import mesh_label_image, MESH_ROOT

KAPPA_Q = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]      # kappa as a quantile of D over live primitives


def one(scene, recon, n_views, class_set, cap, label_mode, rounds, dev="cuda"):
    from scipy.spatial import cKDTree
    import open3d as o3d
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    wp.init()

    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"output/scannet_{scene}_{recon}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"output/scannet_{scene}_{recon}/model.pt")
    centers = model.points.detach().cpu().numpy(); P = centers.shape[0]
    adj = model.adjacency.detach().long().to(dev)
    aoff = model.adjacency_offsets.detach().long().to(dev)
    src = torch.repeat_interleave(torch.arange(P, device=dev), aoff.diff())   # owner of each edge

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(q)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(all_names)}
    pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if n2i[n] in pres]
    C = len(kept)
    gt_lab = remap_gt_labels(raw, [n2i[n] for n in kept])
    lab_pts, lab_cls = gt_pts[gt_lab > 0], gt_lab[gt_lab > 0]
    T = embed_class_names(kept, dev); TT = T @ T.T

    mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_ROOT, scene, "points3d.ply"))
    tri = np.asarray(mesh.triangles); vert_cls = gt_lab.astype(np.int64)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cache_dir = os.path.join("artifacts", "scannet", scene, "gtlabels")
    gt_tree_all = cKDTree(gt_pts); gt_cls_all = gt_lab.astype(np.int64)

    ncam = len(dh.cameras)
    sel = list(range(ncam)) if n_views <= 0 else \
        np.linspace(0, ncam - 1, min(n_views, ncam)).astype(int).tolist()

    AtS = torch.zeros(P, C, device=dev); D = torch.zeros(P, device=dev)
    for vi in sel:
        cam = dh.cameras[vi]; H, W = int(cam.height), int(cam.width)
        c2w = torch.eye(4, dtype=torch.float64); c2w[:3, :4] = dh.c2ws[vi].double()
        cls_img = mesh_label_image(scene, vi, cam, c2w.float(), H, W, rc, tri, vert_cls,
                                   dev, cache_dir, label_mode, gt_tree_all, gt_cls_all)
        if not bool((cls_img > 0).any()):
            del cls_img; continue
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        r_ = op.row_indices.to(torch.int64).to(dev)
        c_ = op.col_indices.to(torch.int64).to(dev)
        v_ = op.values.float().to(dev)
        del op
        keep = cls_img[r_] > 0
        r_, c_, v_ = r_[keep], c_[keep], v_[keep]
        if r_.numel():
            AtS.index_put_((c_, cls_img[r_] - 1), v_, accumulate=True)
            D.index_add_(0, c_, v_)
        del cls_img, r_, c_, v_, keep
        torch.cuda.empty_cache()

    live = D > 0
    Y0 = torch.zeros(P, C, device=dev); Y0[live] = AtS[live] / D[live].unsqueeze(-1)

    li = np.nonzero(live.cpu().numpy())[0]
    _, loc = cKDTree(centers[li]).query(lab_pts, k=1, workers=-1)
    owner = torch.from_numpy(li[loc].astype(np.int64)).to(dev)
    pt_gt = torch.from_numpy(lab_cls.astype(np.int64)).to(dev)

    def pt_miou(Ym):
        pr = torch.zeros(P, dtype=torch.long, device=dev)
        pr[live] = (Ym[live] @ TT).argmax(1) + 1
        _, mi, _, _ = calculate_metrics(pt_gt.cpu(), pr[owner].cpu(), C + 1)
        return float(mi)

    Dl = D[live]
    out = dict(scene=scene, recon=recon, P=int(P), live=int(live.sum()), C=C, rounds=rounds,
               pt_eq7=pt_miou(Y0))
    for q in KAPPA_Q:
        kap = float(torch.quantile(Dl, q)) if q > 0 else 0.0
        S_, D_ = AtS.clone(), D.clone()
        for _ in range(max(rounds, 1)):
            # neighbour consensus: sum of neighbours' evidence, normalised by their total weight
            nS = torch.zeros(P, C, device=dev).index_add_(0, src, S_[adj])
            nD = torch.zeros(P, device=dev).index_add_(0, src, D_[adj])
            # A primitive with NO live neighbour has no consensus to shrink toward. Leaving m = 0
            # there would shrink it toward the zero vector, so its row sums to D/(D+kappa) < 1 and
            # the "stays in the simplex" property fails (caught by the runtime check: the residual
            # read 1.0). The argmax happens to be unaffected -- the row is just Y0 scaled by a
            # positive constant -- but the honest fix is to not shrink what has nothing to shrink
            # toward, i.e. fall back to the primitive's own distribution.
            m = torch.zeros(P, C, device=dev)
            ok = nD > 0
            m[ok] = nS[ok] / nD[ok].unsqueeze(-1)
            lone = live & ~ok
            if bool(lone.any()):
                m[lone] = S_[lone] / D_[lone].clamp_min(1e-30).unsqueeze(-1)
            Ysh = torch.zeros(P, C, device=dev)
            Ysh[live] = (S_[live] + kap * m[live]) / (D_[live] + kap).unsqueeze(-1)
            S_, D_ = Ysh * (D_ + kap).unsqueeze(-1), D_ + kap    # keep (evidence, mass) consistent
        if q == 0.0:
            # kappa = 0 must reproduce Eq. 6 exactly, or the sweep is measuring an artefact
            err = float((Ysh[live] - Y0[live]).abs().max())
            assert err < 1e-5, f"kappa=0 does not reproduce Eq.6 (max dev {err:.3e})"
            out["kappa0_identity_err"] = err
        simplex_err = float((Ysh[live].sum(1) - 1.0).abs().max())
        out[f"pt_kappa{q:g}"] = pt_miou(Ysh)
        out[f"kappa{q:g}_value"] = kap
        out[f"kappa{q:g}_simplex_err"] = simplex_err
        del S_, D_, Ysh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=-1)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--label-mode", default="vertex")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default="artifacts/scannet/oracle_shrink.json")
    a = ap.parse_args()
    rows = []
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                r = one(sc, rec, a.views, a.class_set, a.cap, a.label_mode, a.rounds)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache(); continue
            rows.append(r); json.dump(rows, open(a.out, "w"), indent=1)
            best = max(KAPPA_Q, key=lambda q: r[f"pt_kappa{q:g}"])
            print(f"[{rec}/{sc}] eq7 {r['pt_eq7']*100:6.2f}  best kappa=q{best:g} "
                  f"-> {r[f'pt_kappa{best:g}']*100:6.2f} "
                  f"({(r[f'pt_kappa{best:g}']-r['pt_eq7'])*100:+.2f})", flush=True)
            torch.cuda.empty_cache()
    if not rows:
        return
    f = lambda k: float(np.mean([r[k] for r in rows]))
    base = f("pt_eq7")
    print("")
    print(f"=== {len(rows)} scenes, rounds={a.rounds}, per-point mIoU "
          f"(Eq6 baseline {base*100:.2f}) ===")
    print(f"{'kappa (quantile of D)':<24}{'mIoU':>8}{'delta':>8}{'wins':>8}")
    for q in KAPPA_Q:
        v = f(f"pt_kappa{q:g}")
        w = sum(1 for r in rows if r[f"pt_kappa{q:g}"] > r["pt_eq7"])
        print(f"{('q=' + f'{q:g}'):<24}{v*100:>8.2f}{(v-base)*100:>+8.2f}{w:>5}/{len(rows)}")
    print(f"\n  kappa=0 identity check: max deviation from Eq.6 = "
          f"{max(r.get('kappa0_identity_err', 0.0) for r in rows):.2e}")
    print(f"  simplex residual (max |sum-1|): "
          f"{max(r[f'kappa{q:g}_simplex_err'] for r in rows for q in KAPPA_Q):.2e}")


if __name__ == "__main__":
    main()
