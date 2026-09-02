"""Analysis half of diagnose_lifting_rays.py: which LIFTING-stage ray statistics predict
per-cell correctness, and which do not.

Reports, per scene and pooled:
  * distributions of the column sum A^T 1, the ray count per cell, the view count per cell
  * ray-side: how many DISTINCT GT objects a ray deposits into, and the dominant-object share
  * per-cell contamination: share of incoming weight from rays whose dominant object differs
  * depth-order statistics: mean traversal slot, mean transmittance-on-arrival, first-hit share
  * AUC of each statistic against per-cell correctness, RAW and PARTIALLED on reliability
    (decile-of-statistic accuracy computed WITHIN reliability quintiles)
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, remap_gt_labels, embed_class_names

SCR = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad"
DCACHE = os.path.join(SCR, "dcache")
SCENES = ["scene0347_00", "scene0070_00", "scene0140_00", "scene0645_00", "scene0590_00",
          "scene0062_00", "scene0000_00", "scene0097_00", "scene0200_00", "scene0400_00"]


def auc(score, label):
    """Mann-Whitney AUC, ties averaged."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    y = label[order].astype(np.float64)
    rank = np.empty_like(s, dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_scene(scene, device="cuda"):
    d = np.load(os.path.join(SCR, f"rays_{scene}.npz"), allow_pickle=True)
    solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_ogl3.pt",
                        map_location="cpu", weights_only=True)
    valid_mask = solved["valid_mask"].cpu().numpy()
    valid_idx = np.where(valid_mask)[0]
    c = torch.load(os.path.join(DCACHE, f"{scene}_ogl3.pt"), map_location="cpu", weights_only=False)
    unit = F.normalize(c["unit"].to(device).float(), dim=-1)
    raw, prow, names = c["raw_labels"].numpy(), c["point_row"].numpy(), c["all_names"]
    n2i = {n: i for i, n in enumerate(names)}
    present = set(np.unique(raw).tolist())
    kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
    gt = remap_gt_labels(raw, [i for i, _ in kept])
    text = embed_class_names([n for _, n in kept], device)
    pred = (unit @ text.T).argmax(-1).cpu().numpy() + 1
    nC = len(kept) + 1
    hist = np.zeros((unit.shape[0], nC), dtype=np.int32)
    ok = (prow >= 0) & (gt > 0)
    np.add.at(hist, (prow[ok], gt[ok]), 1)
    scorable_rows = hist.sum(1) > 0
    maj = hist.argmax(1)
    correct_rows = (pred == maj)

    # per-primitive stats file (reliability)
    st = torch.load(f"artifacts/scannet/{scene}/stats_nonfrozen_ogl3.pt",
                    map_location="cpu", weights_only=False)
    sup = st["support"].numpy()
    nn = st["numerator"].norm(dim=-1).numpy()
    intra = np.maximum(st["intra_sum"].numpy(), 1e-12)
    n_eff_v = sup ** 2 / np.maximum(st["sum_view_weight_sq"].numpy(), 1e-12)
    norm_f = nn / np.maximum(sup, 1e-12)
    rel = norm_f * n_eff_v / (n_eff_v + 1.0)

    P = sup.shape[0]
    g = {k: d[k] for k in d.files if d[k].shape == (P,)}
    eps = 1e-12
    ws = np.maximum(g["w_sum"], eps)
    feat = {
        "support(A^T 1)": g["w_sum"],
        "n_rays": g["n_rays"],
        "n_views": g["n_views"],
        "reliability": rel,
        "n_eff_views": n_eff_v,
        "mean_slot": g["w_slot"] / ws,
        "mean_trans_on_arrival": g["w_trans"] / ws,
        "first_hit_share": g["w_first"] / ws,
        "firstsig_share": g["w_firstsig"] / ws,
        "front05_share": g["w_front05"] / ws,
        "sam_purity": g["sam_top"] / np.maximum(g["sam_tot"], eps),
        "sam_groups_per_view": g["sam_groups"] / np.maximum(g["n_views"], 1),
        "ray_contam_share": g["w_contam"] / np.maximum(g["w_labelled"], eps),
        "w_per_ray": g["w_sum"] / np.maximum(g["n_rays"], 1),
        "rays_per_view": g["n_rays"] / np.maximum(g["n_views"], 1),
    }
    # rows -> global primitive index
    scorable_glob = valid_idx[scorable_rows]
    correct = correct_rows[scorable_rows]
    majority_class = np.zeros(P, dtype=np.int64)
    majority_class[valid_idx] = maj
    out = {"majority_class": majority_class,
           "class_names": [n for _, n in kept],
           "scene": scene, "P": P, "n_valid": len(valid_idx),
           "n_scorable": int(scorable_rows.sum()), "acc": float(correct.mean()),
           "feat": {k: v[scorable_glob] for k, v in feat.items()},
           "cls": majority_class[scorable_glob],
           "correct": correct,
           "global": {k: v for k, v in feat.items()},
           "valid_mask": valid_mask,
           "gtlab": d["gtlab"],
           "ray_ncls": d["ray_ncls"], "ray_domshare_hist": d["ray_domshare_hist"],
           "ray_nnz_hist": d["ray_nnz_hist"], "slot_w_hist": d["slot_w_hist"],
           "n_rays_total": float(d["n_rays_total"][0]),
           "w_contam_tot": float(d["w_contam"].sum()), "w_lab_tot": float(d["w_labelled"].sum())}
    return out


def main():
    only = [s for s in os.environ.get("ONLY", "").split(",") if s]
    scenes = only or [s for s in SCENES if os.path.exists(os.path.join(SCR, f"rays_{s}.npz"))]
    res = [per_scene(s) for s in scenes]

    print("\n===== 1. COVERAGE / RAY COUNT PER CELL =====")
    print(f"{'scene':<14}{'P':>8}{'valid%':>8}{'rays/cell p10':>14}{'p50':>8}{'p90':>9}"
          f"{'views/cell p50':>15}{'support p50':>12}{'acc':>7}")
    for r in res:
        gl = r["global"]
        v = r["valid_mask"]
        nr, nv, sp = gl["n_rays"][v], gl["n_views"][v], gl["support(A^T 1)"][v]
        print(f"{r['scene']:<14}{r['P']:>8}{100*v.mean():>8.1f}"
              f"{np.percentile(nr,10):>14.0f}{np.percentile(nr,50):>8.0f}{np.percentile(nr,90):>9.0f}"
              f"{np.percentile(nv,50):>15.0f}{np.percentile(sp,50):>12.2f}{r['acc']:>7.4f}")

    print("\n----- ray-count decile vs per-cell accuracy (scorable cells) -----")
    for key in ["n_rays", "n_views", "support(A^T 1)", "reliability"]:
        rows = []
        for r in res:
            x, y = r["feat"][key], r["correct"]
            q = np.quantile(x, np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(q[1:-1], x), 0, 9)
            rows.append([y[dec == dd].mean() if (dec == dd).any() else np.nan for dd in range(10)])
        m = np.nanmean(np.array(rows), 0)
        A = np.nanmean([auc(r["feat"][key], r["correct"]) for r in res])
        print(f"{key:<18} AUC {A:.4f}  deciles " + " ".join(f"{a:.3f}" for a in m))

    print("\n===== 2. CONTAMINATION =====")
    print(f"{'scene':<14}{'rays w/ GT':>12}{'1 class':>9}{'2':>8}{'3':>8}{'>=4':>8}"
          f"{'domshare p50':>13}{'cell contam wshare':>20}")
    for r in res:
        h = r["ray_ncls"]
        tot = h.sum()
        cum = np.cumsum(r["ray_domshare_hist"]) / max(r["ray_domshare_hist"].sum(), 1)
        p50 = (np.searchsorted(cum, 0.5) + 0.5) / 20.0
        print(f"{r['scene']:<14}{tot:>12.3e}{100*h[1]/tot:>8.1f}%{100*h[2]/tot:>7.1f}%"
              f"{100*h[3]/tot:>7.1f}%{100*h[4:].sum()/tot:>7.1f}%{p50:>13.3f}"
              f"{100*r['w_contam_tot']/max(r['w_lab_tot'],1e-9):>19.1f}%")

    print("\n----- contamination / purity statistics vs correctness -----")
    for key in ["ray_contam_share", "sam_purity", "sam_groups_per_view"]:
        rows, aucs = [], []
        for r in res:
            x, y = r["feat"][key], r["correct"]
            q = np.quantile(x, np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(q[1:-1], x), 0, 9)
            rows.append([y[dec == dd].mean() if (dec == dd).any() else np.nan for dd in range(10)])
            aucs.append(auc(x, y))
        m = np.nanmean(np.array(rows), 0)
        print(f"{key:<20} AUC {np.nanmean(aucs):.4f}  deciles " + " ".join(f"{a:.3f}" for a in m))

    print("\n===== 3. DEPTH ORDER ALONG THE RAY =====")
    print("pooled weight by traversal slot (share of total A mass):")
    sw = np.sum([r["slot_w_hist"] for r in res], 0)
    sw = sw / sw.sum()
    print("  slot " + " ".join(f"{i}:{sw[i]*100:5.1f}%" for i in range(8)) + f"  8+:{sw[8:].sum()*100:.1f}%")
    print("pooled nonzeros per ray:")
    rn = np.sum([r["ray_nnz_hist"] for r in res], 0)
    rn = rn / rn.sum()
    print("  nnz " + " ".join(f"{i}:{rn[i]*100:5.1f}%" for i in range(9)) + f"  9+:{rn[9:].sum()*100:.1f}%")
    for key in ["mean_slot", "mean_trans_on_arrival", "first_hit_share", "firstsig_share",
                "front05_share", "w_per_ray", "rays_per_view", "n_eff_views"]:
        rows, aucs = [], []
        for r in res:
            x, y = r["feat"][key], r["correct"]
            q = np.quantile(x, np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(q[1:-1], x), 0, 9)
            rows.append([y[dec == dd].mean() if (dec == dd).any() else np.nan for dd in range(10)])
            aucs.append(auc(x, y))
        m = np.nanmean(np.array(rows), 0)
        print(f"{key:<22} AUC {np.nanmean(aucs):.4f}  deciles " + " ".join(f"{a:.3f}" for a in m))

    print("\n===== 4. PARTIALLED ON RELIABILITY (decile of stat WITHIN reliability quintile) =====")
    for key in ["n_rays", "front05_share", "sam_purity", "ray_contam_share",
                "mean_trans_on_arrival", "n_views"]:
        tab = np.full((5, 5), np.nan)
        cnt = np.zeros((5, 5))
        for r in res:
            rel, x, y = r["feat"]["reliability"], r["feat"][key], r["correct"]
            rq = np.clip(np.searchsorted(np.quantile(rel, [.2, .4, .6, .8]), rel), 0, 4)
            for i in range(5):
                m = rq == i
                if m.sum() < 50:
                    continue
                xq = np.clip(np.searchsorted(np.quantile(x[m], [.2, .4, .6, .8]), x[m]), 0, 4)
                for j in range(5):
                    s = m.copy()
                    s[m] = xq == j
                    if s.sum():
                        v = y[s].mean() * s.sum()
                        tab[i, j] = (0 if np.isnan(tab[i, j]) else tab[i, j]) + v
                        cnt[i, j] += s.sum()
        tab = tab / np.maximum(cnt, 1)
        spread = np.nanmean(tab[:, 4] - tab[:, 0])
        print(f"\n{key}: within-reliability-quintile accuracy by {key} quintile "
              f"(mean Q5-Q1 = {spread:+.4f})")
        for i in range(5):
            print(f"  rel Q{i+1}: " + " ".join(f"{tab[i,j]:.3f}" for j in range(5)))


    print("\n===== 5. WITHIN-GT-CLASS AUC (kills the class-composition confound) =====")
    keys = ["n_rays", "n_views", "support(A^T 1)", "reliability", "mean_slot",
            "mean_trans_on_arrival", "first_hit_share", "front05_share", "sam_purity",
            "sam_groups_per_view", "ray_contam_share", "w_per_ray", "n_eff_views"]
    print(f"{'stat':<24}{'raw AUC':>10}{'within-class AUC':>19}{'n cells used':>14}")
    for key in keys:
        num, den, raw = 0.0, 0.0, []
        for r in res:
            raw.append(auc(r["feat"][key], r["correct"]))
            for cl in np.unique(r["cls"]):
                m = r["cls"] == cl
                if m.sum() < 200:
                    continue
                y = r["correct"][m]
                if y.mean() in (0.0, 1.0):
                    continue
                a = auc(r["feat"][key][m], y)
                if not np.isnan(a):
                    num += a * m.sum()
                    den += m.sum()
        print(f"{key:<24}{np.nanmean(raw):>10.4f}{num/max(den,1):>19.4f}{int(den):>14}")

    print("\n===== 6. COVERAGE-GATE SIMULATION (what a gate would remove) =====")
    for key, lo in [("sam_purity", True), ("n_rays", False), ("mean_slot", False),
                    ("front05_share", True), ("reliability", True)]:
        print(f"\ngate on {key} (drop {'lowest' if lo else 'highest'}):")
        print(f"  {'drop%':>7}{'acc dropped':>13}{'acc kept':>10}{'acc all':>9}"
              f"{'GT pts dropped%':>17}")
        for frac in [0.05, 0.10, 0.20, 0.30]:
            ad, ak, gp, w = [], [], [], []
            for r in res:
                x, y = r["feat"][key], r["correct"]
                thr = np.quantile(x, frac if lo else 1 - frac)
                drop = x <= thr if lo else x >= thr
                if drop.sum() == 0 or (~drop).sum() == 0:
                    continue
                ad.append(y[drop].mean()); ak.append(y[~drop].mean())
                gp.append(drop.mean()); w.append(y.mean())
            print(f"  {100*np.mean(gp):>6.1f}%{np.mean(ad):>13.4f}{np.mean(ak):>10.4f}"
                  f"{np.mean(w):>9.4f}{100*np.mean(gp):>16.1f}%")


if __name__ == "__main__":
    main()
