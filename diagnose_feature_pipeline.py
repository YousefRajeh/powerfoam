"""Which STEP of the feature pipeline is actually losing the accuracy?

The oracle diagnostic showed grouping is not binding (per-primitive oracle 93.7-96.5 while we
score 23-39). But it labelled groups from ground truth, so it bypassed the features entirely
and cannot say WHERE in the feature path the loss happens. Three candidates:

  (1) the 2D features        SAM masks + CLIP embeddings are wrong to begin with
  (2) the lifting / solve    2D -> per-primitive aggregation destroys the information
  (3) the text matching      features are fine but cosine-vs-class-name is a bad classifier

These separate cleanly with three measurements on the SAME lifted features:

  TEXT ARGMAX + RANK   what we actually do, plus WHERE the true class sits in the cosine
                       ordering. Rank 2-3 means the feature is nearly right and the failure
                       is calibration; rank 10+ means the feature does not encode the class.

  VISUAL PROTOTYPES    replace the CLIP text embedding of each class with the MEAN LIFTED
                       FEATURE of cells of that class, estimated on half the cells and tested
                       on the other half (so it is honest, not a lookup). This asks: are the
                       features linearly separable by class in their own space? If prototypes
                       work where text fails, the loss is the image-text modality gap, not the
                       features -- and that is a known, attackable CLIP property.

  LINEAR PROBE         logistic regression on the lifted features, train/test split. The
                       upper bound on what ANY linear classifier could extract. If the probe
                       is high, the information IS in the features and every point below it is
                       classifier loss. If the probe is low too, the loss is upstream, in the
                       2D features or the solve, and no amount of prompt engineering helps.

Per-cell ground truth is the majority GT label among the points a cell owns, so only cells
that own labelled points are scored -- the same population the real metric sees.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
                                       remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import SCENES


def cell_gt_labels(gt_labels, assigned, n_cells, n_classes):
    """Majority GT label per cell (0 = no labelled point owned)."""
    owned = assigned >= 0
    counts = np.zeros((n_cells, n_classes), dtype=np.int64)
    np.add.at(counts, (assigned[owned], gt_labels[owned]), 1)
    counts[:, 0] = 0                       # ignore-class never wins
    lab = counts.argmax(1)
    lab[counts.sum(1) == 0] = 0
    return lab, counts


def linear_probe(X, y, n_classes, device, epochs=300, lr=0.05, seed=0):
    """Multinomial logistic regression, half train / half test. Upper bound for a linear
    classifier on these features -- deliberately simple, since the real classifier (cosine
    against a fixed text vector) is itself linear."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    n_tr = X.shape[0] // 2
    tr, te = perm[:n_tr].to(device), perm[n_tr:].to(device)
    W = torch.zeros(X.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    Xtr, ytr = X[tr], y[tr]
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (( X[te] @ W + b).argmax(-1) == y[te]).float().mean().item()
    return acc


def main():
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0347_00,scene0070_00,scene0140_00,scene0645_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    cs = a.class_set
    rows = {}
    for scene in a.scenes.split(","):
        split = SCENES[scene]
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(
            f"{a.gt_root}/{split}/{scene}", "segment20")
        centers, radii = load_foam(f"output/scannet_{scene}_{a.variant}", device)
        solved = torch.load(
            f"artifacts/scannet/{scene}/solved_geometric_median_{a.variant}_l3.pt",
            map_location=device, weights_only=True)
        feats = solved["primitive_features"].to(device).float()
        vm = solved["valid_mask"].cpu().numpy()
        unit = torch.zeros_like(feats)
        vi = torch.where(torch.from_numpy(vm).to(device))[0]
        unit[vi] = F.normalize(feats[vi], dim=-1)

        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = remap_gt_labels(raw_labels, tids)
        n_classes = len(tids) + 1
        text = embed_class_names(tnames, device)          # (K, 512), K = n_classes - 1

        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
        lab, _ = cell_gt_labels(gt_t, assigned, centers.shape[0], n_classes)
        has = (lab > 0) & vm
        idx = torch.from_numpy(np.where(has)[0]).to(device)
        X = unit[idx]                                     # (N, 512) unit features
        y = torch.from_numpy(lab[has] - 1).to(device).long()   # 0-based class ids

        sim = X @ text.T                                  # (N, K)
        pred = sim.argmax(-1)
        acc_text = (pred == y).float().mean().item()
        # rank of the TRUE class in the cosine ordering (1 = correct)
        order = sim.argsort(dim=-1, descending=True)
        rank = (order == y[:, None]).float().argmax(-1) + 1
        # visual prototypes, estimated on half the cells and tested on the other half
        gsplit = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(X.shape[0], generator=gsplit).to(device)
        half = X.shape[0] // 2
        tr, te = perm[:half], perm[half:]
        protos = torch.zeros(len(tids), X.shape[1], device=device)
        cnt = torch.zeros(len(tids), device=device)
        protos.index_add_(0, y[tr], X[tr])
        cnt.index_add_(0, y[tr], torch.ones_like(y[tr], dtype=torch.float))
        alive = cnt > 0
        protos[alive] = F.normalize(protos[alive] / cnt[alive, None], dim=-1)
        acc_proto = ((X[te] @ protos.T).argmax(-1) == y[te]).float().mean().item()
        acc_text_te = ((X[te] @ text.T).argmax(-1) == y[te]).float().mean().item()
        acc_probe = linear_probe(X, y, len(tids), device)

        rows[scene] = {"cells_scored": int(X.shape[0]), "n_classes": len(tids),
                       "acc_text": acc_text, "acc_text_testhalf": acc_text_te,
                       "acc_prototype": acc_proto, "acc_linear_probe": acc_probe,
                       "true_class_rank_mean": float(rank.float().mean()),
                       "true_class_rank_median": float(rank.median()),
                       "top3": float((rank <= 3).float().mean())}
        r = rows[scene]
        print(f"\n=== {scene} ({r['cells_scored']} labelled cells, {r['n_classes']} classes) ===")
        print(f"  text argmax (what we do)      acc = {r['acc_text']*100:5.2f}%")
        print(f"  visual prototypes (half-fit)  acc = {r['acc_prototype']*100:5.2f}%   "
              f"[text on same half: {r['acc_text_testhalf']*100:5.2f}%]")
        print(f"  linear probe (half-fit)       acc = {r['acc_linear_probe']*100:5.2f}%")
        print(f"  true-class rank: mean {r['true_class_rank_mean']:.2f} / "
              f"median {r['true_class_rank_median']:.0f} of {r['n_classes']}   "
              f"top-3 {r['top3']*100:.1f}%")

    print("\nHOW TO READ THIS:")
    print("  probe >> text      -> the information IS in the lifted features; the loss is the")
    print("                        classifier (text embeddings / modality gap), not the solve.")
    print("  prototype >> text  -> specifically the IMAGE-TEXT gap, since prototypes are the")
    print("                        same features compared in their own space.")
    print("  probe ~ text (low) -> the loss is UPSTREAM: 2D features or the lifting/solve.")
    if a.output:
        json.dump(rows, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
