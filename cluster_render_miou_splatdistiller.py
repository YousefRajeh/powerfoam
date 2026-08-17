"""Cluster the solved room0_splatdistiller per-Gaussian features (K=10, spherical
k-means, matching Feature Foam's exact segment.py/segment_cli.py invocation:
num_iters=25, seed=0), render the one-hot cluster assignment onto the 113 held-out
test views (same test_idx = idx % 8 == 0 as everywhere else in this project), and
write a segmentation_*.pt in the exact format evaluate_segmentation_miou.py expects
(rendered_labels, row_view_ids, report.num_clusters) -- so the SAME mIoU script
already used for Feature Foam and gsplat_baseline can be reused unmodified.
"""
import sys
sys.path.insert(0, r"D:\Downloads\splat-distiller")
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from feature_foam_lifting.segment import spherical_kmeans

from gsplat_ext import Parser, Dataset, GaussianPrimitive, GaussianRenderer

CKPT = r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\ckpts\ckpt_29999_rank0.pt"
FEATURES = r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\ckpts\ckpt_29999_rank0_features.pt"
DATA_DIR = r"D:\Downloads\powerfoam\data\replica\room_0_colmap"
OUT = Path(r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\segmentation_splatdistiller_samfeatures_k10.pt")
NUM_CLUSTERS = 10
NUM_ITERS = 25
SEED = 0

features = torch.load(FEATURES, map_location="cuda", weights_only=True)
valid = features.abs().sum(-1) > 0
print(f"primitives: {features.shape[0]}, valid: {int(valid.sum())}")

assignment, centroids = spherical_kmeans(features, valid, NUM_CLUSTERS, num_iters=NUM_ITERS, seed=SEED)
cluster_sizes = torch.bincount(assignment[assignment >= 0], minlength=NUM_CLUSTERS)
print("cluster_sizes:", cluster_sizes.tolist())

splats = GaussianPrimitive()
splats.from_file(CKPT)
splats.to("cuda")

one_hot = F.one_hot(assignment.clamp_min(0), num_classes=NUM_CLUSTERS).float()
one_hot[assignment < 0] = 0.0
splats._feature = one_hot.cuda()
renderer = GaussianRenderer(splats)

parser = Parser(data_dir=DATA_DIR, factor=1, test_every=8)
valset = Dataset(parser, split="val")
valLoader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
print(f"num test views: {len(valset)}")

all_labels = []
all_row_view_ids = []
unassigned = 0
total = 0
for view_idx, data in enumerate(tqdm(valLoader, desc="Rendering cluster maps on test views")):
    camtoworlds = data["camtoworld"].to("cuda")
    Ks = data["K"].to("cuda")
    pixels = data["image"]
    height, width = pixels.shape[1:3]

    rendered = renderer.render(K=Ks, extrinsic=camtoworlds, width=width, height=height, mode="Feature")
    # rendered: (H, W, num_clusters) mixture -- alpha-blended, doesn't sum to 1 where
    # background shows through, so treat near-zero-total pixels as unassigned (-1)
    # exactly like render_cluster_map does for the sparse-operator path.
    total_weight = rendered.sum(dim=-1)
    label_dense = rendered.argmax(dim=-1)
    label_dense[total_weight <= 1e-6] = -1

    unassigned += int((label_dense < 0).sum())
    total += label_dense.numel()
    all_labels.append(label_dense.reshape(-1))
    all_row_view_ids.append(torch.full((label_dense.numel(),), view_idx, dtype=torch.long))

rendered_labels = torch.cat(all_labels)
row_view_ids = torch.cat(all_row_view_ids)

report = {
    "num_clusters": NUM_CLUSTERS, "num_iters": NUM_ITERS, "seed": SEED,
    "cluster_sizes": cluster_sizes.tolist(), "num_unassigned": int((assignment < 0).sum()),
    "rendered_unassigned_fraction": unassigned / total,
}
save_dict = {
    "assignment": assignment.cpu(), "centroids": centroids.cpu().to(torch.float16),
    "report": report, "rendered_labels": rendered_labels.cpu(), "row_view_ids": row_view_ids.cpu(),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(save_dict, OUT)
OUT.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(f"wrote {OUT}")
print(json.dumps(report, indent=2))
