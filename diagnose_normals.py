"""Why did plane_first find no planes? Measure, on scene0000_00's actual wall/floor
primitives (identified via GT point ownership):
  1. Trained-normal coherence: angular spread of model.get_normals() among wall
     primitives and among floor primitives (should be ~0 deg spread for a real plane).
  2. Local-PCA normal coherence: normals re-estimated from each primitive's adjacency
     neighborhood positions (smallest eigenvector of the local covariance) -- if these
     are much tighter, the trained quaternions are the problem, not the plane gates.
  3. Coplanarity scale: distribution of out-of-plane offsets of wall-primitive centers
     from the best-fit wall plane -- calibrates coplanar_eps against real cell jitter.
"""
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import load_scannet_pointcept_gt
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam

device = "cuda"
scene = "scene0000_00"
gt_dir = rf"D:\Downloads\scannet_pointcept\train\{scene}"
ckpt_dir = f"output/scannet_{scene}_nonfrozen"

gt_points, raw_labels, all_names = load_scannet_pointcept_gt(gt_dir, "segment20")
centers, radii, normals_np = load_foam(ckpt_dir, device, return_normals=True)
solved = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen.pt",
                    map_location=device, weights_only=True)
valid_mask = solved["valid_mask"].cpu().numpy()
adj = torch.load(f"artifacts/scannet/{scene}/adjacency_nonfrozen.pt",
                 map_location=device, weights_only=True)
adjacent, offsets = adj["adjacent"].to(device).long(), adj["offsets"].to(device).long()

assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)

positions = torch.from_numpy(centers).to(device).float()
tn = F.normalize(torch.from_numpy(normals_np).to(device).float(), dim=-1)

# local-PCA normals from adjacency neighborhoods
P = positions.shape[0]
src = torch.repeat_interleave(torch.arange(P, device=device), (offsets[1:] - offsets[:-1]))
rel = positions[adjacent] - positions[src]
cov = torch.zeros(P, 3, 3, device=device)
cov.index_add_(0, src, rel.unsqueeze(-1) * rel.unsqueeze(-2))
deg = (offsets[1:] - offsets[:-1]).clamp_min(1).float()
cov /= deg[:, None, None]
# cusolver's batched eigh chokes at this batch size; tiny 3x3 problems run fine on CPU
evals, evecs = torch.linalg.eigh(cov.cpu().double())
evals, evecs = evals.to(device).float(), evecs.to(device).float()
pca_n = evecs[..., 0]
planarity = 1.0 - evals[:, 0] / evals[:, 1].clamp_min(1e-12)  # ~1 for plane-like nbhd

wall_id = all_names.index("wall")
floor_id = all_names.index("floor")
for cls_name, cls_id in (("wall", wall_id), ("floor", floor_id)):
    pts = np.where(raw_labels == cls_id)[0]
    prims = np.unique(assigned[pts][assigned[pts] >= 0])
    prims_t = torch.from_numpy(prims).to(device)
    print(f"\n=== {cls_name}: {len(pts)} GT points -> {len(prims)} owning primitives ===")
    for tag, nrm in (("trained", tn), ("local-PCA", pca_n)):
        n = nrm[prims_t]
        # sign-align to the dominant direction before measuring spread
        ref = n[0]
        sgn = torch.sign(n @ ref).unsqueeze(1)
        sgn[sgn == 0] = 1
        mean_n = F.normalize((n * sgn).mean(0), dim=0)
        cos = (n @ mean_n).abs()
        ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
        print(f"  {tag:>10} normals: |cos| to mean p50={cos.median():.3f} "
              f"p10={cos.quantile(0.1):.3f}; angle p50={ang.median():.1f}deg p90={ang.quantile(0.9):.1f}deg")
    # out-of-plane offsets vs best-fit plane through these primitives (PCA of positions)
    pos = positions[prims_t]
    pc = pos - pos.mean(0)
    _, _, V = torch.linalg.svd(pc, full_matrices=False)
    plane_n = V[2]
    d = (pc @ plane_n).abs()
    print(f"  center offsets from best-fit plane: p50={d.median()*100:.1f}cm "
          f"p90={d.quantile(0.9)*100:.1f}cm p99={d.quantile(0.99)*100:.1f}cm")
    print(f"  local planarity score (1=planar nbhd): p50={planarity[prims_t].median():.3f}")
