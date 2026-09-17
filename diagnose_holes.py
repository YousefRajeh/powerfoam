"""WHERE DO THE HOLES COME FROM? -- geometry, renderer, solve, or the upstream model.

The theorem says the closed form is exact when rays are disjoint. Foam rays are near one-hot,
so the SOLVE is essentially exact and cannot be what is wrong. But exactness is a statement
about BIAS, not VARIANCE: under disjoint support a primitive is estimated from ITS OWN rays
alone, with no borrowing from neighbours. The failure mode therefore moves from "wrong answer"
to "no answer" -- which is exactly what a hole is.

Three per-primitive quantities, all ALREADY in the accumulated stats, separate the causes:

  D_jj = support     total ray weight the primitive ever received.  D=0 -> nothing to lift onto.
  G_jj = support2    diag(A^T A).  c_j = G_jj/D_jj in (0,1] is the theorem's concentration:
                     1 means every ray touching j was OWNED by j (the disjoint limit).
  R_j = ||numerator|| / intra_sum    resultant length of the per-view features.
                     1 = every contributing view agrees, 0 = they cancel.

Attribution of each scored GT point:
  dead      assigned primitive has D_jj = 0, or the point is owned by no cell  -> geometry
  culled    assigned primitive dropped by the opacity threshold               -> renderer/opacity
  starved   D_jj in the bottom decile                                         -> solve variance
  conflict  R_j in the bottom decile                                          -> view disagreement
  upstream  well-supported, views agree, still wrong                          -> CLIP itself
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, r"D:\Downloads\powerfoam")
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels)
from diagnose_scannet_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells

GT_ROOT = r"D:\Downloads\scannet_pointcept"
SCENES = ["scene0000_00","scene0062_00","scene0070_00","scene0097_00","scene0140_00",
          "scene0200_00","scene0347_00","scene0400_00","scene0590_00","scene0645_00"]


def geometry(scene, recon):
    """centers / radii / density from the checkpoint, cached -- loading the model is the slow part."""
    cache = f"artifacts/scannet/{scene}/geom_{recon}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["centers"], z["radii"], z["density"]
    import warp as wp, configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    wp.init()
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    m = PowerfoamScene(args); m.initialize_from_dataset(dh, device="cuda")
    m.load_pt(f"{ck}/model.pt")
    centers = m.points.detach().cpu().numpy()
    radii = m.get_radii().detach().cpu().numpy()
    density = m.get_density().detach().float().cpu().numpy().reshape(-1)
    np.savez(cache, centers=centers, radii=radii, density=density)
    return centers, radii, density


def one_scene(scene, recon, class_set, opacity_threshold, dev="cuda"):
    ap = f"artifacts/scannet/{scene}"
    st = torch.load(f"{ap}/stats_{recon}_ogl3.pt", map_location="cpu", weights_only=False)
    sol = torch.load(f"{ap}/solved_geometric_median_{recon}_ogl3.pt", map_location="cpu",
                     weights_only=True)
    D = st["support"].numpy().astype(np.float64)
    G = st["support2"].numpy().astype(np.float64)
    R = (st["numerator"].norm(dim=-1).numpy().astype(np.float64) /
         np.maximum(st["intra_sum"].numpy().astype(np.float64), 1e-12))
    conc = np.where(D > 0, G / np.maximum(D, 1e-12), np.nan)
    svw = st["sum_view_weight_sq"].numpy().astype(np.float64)
    n_eff = np.where(svw > 0, D ** 2 / np.maximum(svw, 1e-12), 0.0)
    feats = sol["primitive_features"].to(dev).float()
    valid = sol["valid_mask"].numpy()

    centers, radii, density = geometry(scene, recon)
    alpha = 1.0 - np.exp(-density * radii.reshape(-1) * 2.0)
    culled = alpha < opacity_threshold

    cand = [p for p in glob.glob(os.path.join(GT_ROOT, "*", scene)) if os.path.isdir(p)]
    gt_pts, raw, all_names = load_scannet_pointcept_gt(cand[0], "segment20")
    name_to_id = {n: i for i, n in enumerate(all_names)}
    present = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[class_set] if name_to_id[n] in present]
    gt_lab = remap_gt_labels(raw, [name_to_id[n] for n in kept])       # 0 = ignore

    text = embed_class_names(kept, dev)
    pred_cls = (F.normalize(feats, dim=-1) @ text.T).argmax(1).cpu().numpy()

    # Assignment uses the SAME candidate set as the reported protocol (valid cells only).
    assigned = assign_points_to_power_cells(gt_pts, centers, radii, valid=valid, k=64)
    return dict(D=D, G=G, R=R, conc=conc, n_eff=n_eff, valid=valid, culled=culled, pred=pred_cls,
                assigned=assigned, gt_lab=gt_lab, kept=kept, alpha=alpha, P=len(D))


def attribute(s, d_lo, r_lo):
    """Split the SCORED GT points (gt_lab > 0) into the five causes above."""
    gt, asg = s["gt_lab"], s["assigned"]
    scored = gt > 0
    owned = scored & (asg >= 0)
    j = np.where(asg >= 0, asg, 0)
    correct = owned & (s["pred"][j] + 1 == gt)

    dead = scored & ((asg < 0) | ((asg >= 0) & (s["D"][j] <= 0)))
    wrong = owned & ~correct & ~dead
    culled = wrong & s["culled"][j]
    rest = wrong & ~culled
    starved = rest & (s["D"][j] < d_lo)
    conflict = rest & ~starved & (s["R"][j] < r_lo)
    upstream = rest & ~starved & ~conflict
    n = int(scored.sum())
    return dict(n=n, acc=float(correct.sum()) / max(n, 1),
                dead=int(dead.sum()), culled=int(culled.sum()), starved=int(starved.sum()),
                conflict=int(conflict.sum()), upstream=int(upstream.sum()))


def deciles(s, key, nb=5):
    """Accuracy of the scored GT points as a function of a per-primitive quantity."""
    gt, asg = s["gt_lab"], s["assigned"]
    m = (gt > 0) & (asg >= 0)
    j = asg[m]
    ok = (s["pred"][j] + 1 == gt[m])
    v = s[key][j]
    fin = np.isfinite(v)
    j, ok, v = j[fin], ok[fin], v[fin]
    qs = np.quantile(v, np.linspace(0, 1, nb + 1))
    out = []
    for b in range(nb):
        sel = (v >= qs[b]) & (v <= qs[b + 1] if b == nb - 1 else v < qs[b + 1])
        out.append((float(qs[b]), float(qs[b + 1]), int(sel.sum()),
                    float(ok[sel].mean()) if sel.sum() else float("nan")))
    return out


def joint_table(s, nb=3):
    """accuracy in the (n_eff, R_j) grid -- R_j alone is confounded by how many views were seen."""
    gt, asg = s["gt_lab"], s["assigned"]
    m = (gt > 0) & (asg >= 0)
    j = asg[m]
    ok = (s["pred"][j] + 1 == gt[m])
    a1, a2 = s["n_eff"][j], s["R"][j]
    q1 = np.quantile(a1, np.linspace(0, 1, nb + 1)); q2 = np.quantile(a2, np.linspace(0, 1, nb + 1))
    out = []
    for i in range(nb):
        row = []
        s1 = (a1 >= q1[i]) & (a1 <= q1[i + 1] if i == nb - 1 else a1 < q1[i + 1])
        for k in range(nb):
            s2 = s1 & (a2 >= q2[k]) & (a2 <= q2[k + 1] if k == nb - 1 else a2 < q2[k + 1])
            row.append((int(ok[s2].sum()), int(s2.sum())))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out", default="artifacts/scannet/hole_attribution.json")
    a = ap.parse_args()

    joint = []
    rows, agg = [], {k: 0 for k in ["n", "dead", "culled", "starved", "conflict", "upstream"]}
    dec = {"D": [], "conc": [], "R": [], "n_eff": []}
    for sc in a.scenes.split(","):
        try:
            s = one_scene(sc, a.recon, a.class_set, a.opacity_threshold)
        except Exception as e:
            print(f"[{sc}] SKIP {type(e).__name__}: {e}"); continue
        live = s["D"] > 0
        d_lo = float(np.quantile(s["D"][live], 0.10))
        r_lo = float(np.quantile(s["R"][live], 0.10))
        r = attribute(s, d_lo, r_lo); r["scene"] = sc
        r["P"] = s["P"]; r["frac_D0"] = float((~live).mean())
        r["frac_culled"] = float(s["culled"].mean())
        r["conc_med"] = float(np.nanmedian(s["conc"]))
        rows.append(r)
        for k in agg: agg[k] += r[k]
        for k in dec: dec[k].append(deciles(s, k))
        joint.append(joint_table(s))
        print(f"[{sc}] P={s['P']:>6,}  D=0 {r['frac_D0']:.1%}  culled {r['frac_culled']:.1%}  "
              f"conc_med {r['conc_med']:.3f}  acc {r['acc']:.4f}  || dead {r['dead']/r['n']:.1%} "
              f"culled {r['culled']/r['n']:.1%} starved {r['starved']/r['n']:.1%} "
              f"conflict {r['conflict']/r['n']:.1%} upstream {r['upstream']/r['n']:.1%}")

    n = agg["n"]
    print(f"\n=== 10-scene pooled ({n:,} scored GT points) ===")
    for k in ["dead", "culled", "starved", "conflict", "upstream"]:
        print(f"  {k:<9} {agg[k]:>9,}  {agg[k]/n:6.2%}")
    print(f"  {'correct':<9} {n - sum(agg[k] for k in ['dead','culled','starved','conflict','upstream']):>9,}")

    for key, label in [("D", "support D_jj"), ("conc", "concentration G_jj/D_jj"),
                       ("R", "view agreement R_j"), ("n_eff", "effective views n_eff")]:
        print(f"\n--- accuracy vs {label} (per-scene quintiles, mean over scenes) ---")
        arr = np.array([[b[3] for b in sc] for sc in dec[key]], dtype=float)
        cnt = np.array([[b[2] for b in sc] for sc in dec[key]], dtype=float)
        for b in range(arr.shape[1]):
            print(f"  Q{b+1}  acc {np.nanmean(arr[:, b]):.4f}   (n~{cnt[:, b].mean():,.0f}/scene)")

    print("\n--- accuracy by (n_eff tercile) x (R_j tercile), pooled ---")
    print(f"  {'':<12}" + "".join(f"R-T{i+1:<10}" for i in range(3)))
    for i in range(3):
        cells = []
        for k in range(3):
            num = sum(j2[i][k][0] for j2 in joint); den = sum(j2[i][k][1] for j2 in joint)
            cells.append(f"{num/den:.4f}({den/1000:.0f}k)" if den else "  --  ")
        print(f"  n_eff-T{i+1:<4} " + "".join(f"{c:<12}" for c in cells))

    json.dump({"rows": rows, "pooled": agg, "deciles": dec}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
