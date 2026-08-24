"""Follow-up to test_opengaussian_codebook_emulation.py: the attractor statistics over the
FULL 19-name OpenGaussian class list (not just the 7 classes present in scene0347_00), so the
numbers are directly comparable to the reported pathology (spread 0.1325-0.2183,
margin<0.01 for 48.6% of cells, picture winning 22.47%). Question: does the 320-cluster
codebook averaging change the spread/margin/win-distribution at all?
"""
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F

from determinism import enable_determinism
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import spherical_kmeans
from run_cluster_classify_eval import two_level_position_aware

SCENE = "scene0347_00"
K = 320


def stats(unit, text_feats, names, tag):
    sim = F.normalize(unit, dim=-1) @ text_feats.T
    means = sim.mean(dim=0)
    t2 = sim.topk(2, dim=-1).values
    margin = t2[:, 0] - t2[:, 1]
    win = torch.bincount(sim.argmax(dim=-1), minlength=len(names)).float() / len(sim)
    print(f"\n--- {tag} (N={len(sim)}) ---")
    print(f"  mean-cosine {means.min():.4f}..{means.max():.4f} (spread {means.max()-means.min():.4f})"
          f"  argmax-of-means={names[int(means.argmax())]}")
    print(f"  margin mean {margin.mean():.4f} median {margin.median():.4f} frac<0.01 {(margin<0.01).float().mean()*100:.2f}%")
    for i in means.argsort(descending=True).tolist():
        print(f"    {names[i]:<16} meancos {means[i]:.4f}  win {win[i]*100:5.2f}%")


def main():
    enable_determinism()
    torch.set_num_threads(8)
    names = OPENGAUSSIAN_CLASS_SETS["opengaussian19"]
    text_feats = embed_class_names(names, "cpu")
    solved = torch.load(rf"D:\Downloads\powerfoam\artifacts\scannet\{SCENE}\solved_geometric_median_nonfrozen_l3.pt",
                        map_location="cpu", weights_only=True)
    vi = np.where(solved["valid_mask"].numpy())[0]
    unit = F.normalize(solved["primitive_features"].float()[torch.from_numpy(vi)], dim=-1)
    ck = torch.load(rf"D:\Downloads\powerfoam\output\scannet_{SCENE}_nonfrozen\model.pt",
                    map_location="cpu", weights_only=False)
    positions = ck["points"][torch.from_numpy(vi)].float()

    stats(unit, text_feats, names, "per-cell (ours)")
    lab, cent = spherical_kmeans(unit, K, seed=0)
    stats(cent, text_feats, names, "flat-kmeans320 centroids (unit)")
    pooled = torch.zeros(K, 512).index_add_(0, lab, unit)
    cnt = torch.bincount(lab, minlength=K).clamp_min(1).unsqueeze(1).float()
    stats(pooled / cnt, text_feats, names, "flat-kmeans320 mean-pooled")
    plab = two_level_position_aware(positions, unit, seed=0)
    ppooled = torch.zeros(K, 512).index_add_(0, plab, unit)
    pcnt = torch.bincount(plab, minlength=K).clamp_min(1).unsqueeze(1).float()
    nz = torch.bincount(plab, minlength=K) > 0
    stats((ppooled / pcnt)[nz], text_feats, names, "pos-aware 64x5 mean-pooled")


if __name__ == "__main__":
    main()
