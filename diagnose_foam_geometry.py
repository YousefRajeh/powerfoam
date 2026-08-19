"""Foam-geometry hypothesis tests (scene0000_00 nonfrozen L3): probe the structures
unique to PowerFoam -- bubble geometry, facet connectivity, the Laguerre partition,
per-cell multi-view statistics -- and measure how each relates to per-primitive
classification correctness. Each test motivates (or kills) a candidate novel idea for
beating NormLift without templates.

A. Bubble geometry vs correctness: are wrong cells geometrically distinctive
   (radius/power weight, facet degree, center-to-owned-points offset)?
B. Interface vs interior errors: what fraction of errors sit on cells that are
   facet-adjacent to a different-GT-label cell (boundary bubbles)?
C. Facet feature-gradient as a boundary detector: AUC of cross-facet feature cosine
   for predicting GT class boundaries (foundation for min-cut/watershed on the REAL
   cell complex).
D. Reliability quality: is our reliability score actually monotone with correctness
   (NormLift claims theirs is)? Compare vs facet-degree and support as alternatives.
E. Instance connectivity (THE mask-association enabler question): for each GT INSTANCE
   (instance.npy), are its owning cells facet-connected? If instances are ~1 component,
   foam connectivity gives instance-coherent grouping FOR FREE -- the ingredient whose
   absence killed the mask-association idea (OpenGaussian trains ins_feat for this).
"""
import sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from feature_foam_lifting.operator import AccumulatedFeatureStats
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam

device = "cuda"
scene = "scene0000_00"
gt_dir = Path(rf"D:\Downloads\scannet_pointcept\train\{scene}")
gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
instance = np.load(gt_dir / "instance.npy").astype(np.int64)

centers, radii = load_foam(f"output/scannet_{scene}_nonfrozen", device)
solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen_l3.pt",
                    map_location=device, weights_only=True)
feats = solved["primitive_features"].to(device).float()
vm = solved["valid_mask"].cpu().numpy()
vm_t = torch.from_numpy(vm).to(device)
vi = torch.where(vm_t)[0]
unit_full = torch.zeros_like(feats)
unit_full[vi] = F.normalize(feats[vi], dim=-1)

stats = AccumulatedFeatureStats.load(f"artifacts/scannet/{scene}/train_stats_sam_nonfrozen_l3.pt")
rel = stats.reliability()["reliability"].to(device).float()
support = stats.support.to(device).float()
adj = torch.load(f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt", map_location=device, weights_only=True)
adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()
P = centers.shape[0]
deg = (offsets[1:] - offsets[:-1]).float()

assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)

# per-cell majority GT label (19cls remap) + owned count + center offset
n2i = {n: i for i, n in enumerate(all_names)}
present = set(np.unique(raw_labels).tolist())
kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS["opengaussian19"] if n2i[n] in present]
tids = [i for i, _ in kept]; tnames = [n for _, n in kept]
gt19 = remap_gt_labels(raw_labels, tids)  # 0=ignore, 1..K

owned_pts = assigned >= 0
cell_of_pt = assigned[owned_pts]
lbl_of_pt = gt19[owned_pts]
sel = lbl_of_pt > 0
cell_of_pt, lbl_of_pt = cell_of_pt[sel], lbl_of_pt[sel]
pts_xyz = gt_points[owned_pts][sel]

K = len(tids)
vote = np.zeros((P, K + 1), dtype=np.int64)
np.add.at(vote, (cell_of_pt, lbl_of_pt), 1)
cell_label = vote.argmax(1)          # 0 = owns no labeled point
cell_npts = vote.sum(1)
owner_mask = cell_label > 0

# center offset: mean distance from cell center to its owned points
off_sum = np.zeros(P); np.add.at(off_sum, cell_of_pt, np.linalg.norm(pts_xyz - centers[cell_of_pt], axis=1))
center_off = np.where(cell_npts > 0, off_sum / np.maximum(cell_npts, 1), 0)

# per-cell prediction (raw names, plain argmax, no refinement -- clean attribution)
text = embed_class_names(tnames, device)
pred_cls = (unit_full @ text.T).argmax(-1).cpu().numpy() + 1
correct = (pred_cls == cell_label) & owner_mask
oc = owner_mask
print(f"=== {scene}: {oc.sum()} owner cells, per-primitive raw accuracy = {correct[oc].mean():.3f} ===")

# --- A: geometry vs correctness ---
print("\nA) bubble geometry vs correctness (owner cells, median [correct] vs [wrong]):")
for name, arr in [("radius (power wt)", radii), ("facet degree", deg.cpu().numpy()),
                  ("center->points offset (m)", center_off), ("owned points", cell_npts)]:
    a = np.asarray(arr, dtype=np.float64)
    print(f"   {name:>26}: correct={np.median(a[oc & correct]):.4f}  wrong={np.median(a[oc & ~correct]):.4f}")

# --- B: interface vs interior errors ---
src = torch.repeat_interleave(torch.arange(P, device=device), (offsets[1:] - offsets[:-1])).cpu().numpy()
dst = adjacent.cpu().numpy()
both = owner_mask[src] & owner_mask[dst]
diff_edge = both & (cell_label[src] != cell_label[dst])
is_interface = np.zeros(P, dtype=bool)
is_interface[src[diff_edge]] = True
n_int = (oc & is_interface).sum(); n_intr = (oc & ~is_interface).sum()
print(f"\nB) interface bubbles: {n_int} ({n_int/oc.sum()*100:.0f}% of owners)")
print(f"   accuracy interface={correct[oc & is_interface].mean():.3f}  interior={correct[oc & ~is_interface].mean():.3f}")
err_at_interface = (~correct & oc & is_interface).sum() / max((~correct & oc).sum(), 1)
print(f"   fraction of ALL errors sitting on interface bubbles: {err_at_interface*100:.0f}%")

# --- C: facet feature-gradient as boundary detector ---
u = unit_full.cpu().numpy()
esim = np.einsum("ec,ec->e", u[src[both]], u[dst[both]])
same = (cell_label[src[both]] == cell_label[dst[both]])
order = np.argsort(esim)
lab = same[order]
n_pos, n_neg = lab.sum(), (~lab).sum()
ranks = np.arange(1, len(lab) + 1)
auc = (ranks[lab].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
print(f"\nC) facet edge feature-cosine as SAME-label predictor: AUC={auc:.3f} "
      f"(same-label edges: {same.mean()*100:.0f}% of {len(same)} owner-owner facets)")
print(f"   median edge cosine: same={np.median(esim[same]):.3f}  boundary={np.median(esim[~same]):.3f}")

# --- D: reliability quality ---
print("\nD) correctness vs per-cell scores (owner cells, decile accuracy low->high):")
for name, s in [("reliability (ours)", rel.cpu().numpy()), ("support", support.cpu().numpy()),
                ("facet degree", deg.cpu().numpy())]:
    sv = np.asarray(s)[oc]; cv = correct[oc]
    qs = np.quantile(sv, np.linspace(0, 1, 11))
    accs = [cv[(sv >= qs[i]) & (sv <= qs[i+1])].mean() for i in range(10)]
    print(f"   {name:>18}: " + " ".join(f"{a:.2f}" for a in accs))

# --- E: instance connectivity ---
inst_of_pt = instance[owned_pts][sel]
uniq_inst = np.unique(inst_of_pt); uniq_inst = uniq_inst[uniq_inst >= 0]
n_single, n_multi, comps_list, sizes = 0, 0, [], []
adj_np_off = offsets.cpu().numpy(); adj_np = adjacent.cpu().numpy()
for iid in uniq_inst:
    cells = np.unique(cell_of_pt[inst_of_pt == iid])
    if len(cells) < 2:
        n_single += 1; comps_list.append(1); sizes.append(len(cells)); continue
    cs = set(cells.tolist())
    seen, comps = set(), 0
    for c in cells:
        if c in seen: continue
        comps += 1
        stack = [c]; seen.add(c)
        while stack:
            x = stack.pop()
            for nb in adj_np[adj_np_off[x]:adj_np_off[x+1]]:
                if nb in cs and nb not in seen:
                    seen.add(nb); stack.append(nb)
    comps_list.append(comps); sizes.append(len(cells))
    if comps == 1: n_single += 1
    else: n_multi += 1
comps_arr = np.array(comps_list); sizes_arr = np.array(sizes)
big = sizes_arr >= 20
print(f"\nE) GT instances: {len(uniq_inst)} total; facet-CONNECTED as a single component: "
      f"{(comps_arr==1).mean()*100:.0f}% overall, {(comps_arr[big]==1).mean()*100:.0f}% of instances with >=20 cells")
print(f"   components per instance: p50={np.median(comps_arr):.0f} p90={np.percentile(comps_arr,90):.0f}; "
      f"largest-component coverage matters for instance-proposal viability")
