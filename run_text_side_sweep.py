"""Text-side interventions, on cached features. The only axis with real headroom.

MEASURED: our lifted features carry the class information (linear probe 91.9%/91.7% on two
scenes) but cosine-vs-CLIP-text reaches only 46-55%, with the true class at mean rank 2.0 of
7-9 and in the top-3 for 85%. Visual prototypes on the same features reach 76-81%. So the
loss is the image-text comparison, not the solve and not the grouping (whose oracle ceiling
is 93-96%).

WHAT CANNOT WORK, and why it is excluded here: any correction that is CONSTANT ACROSS CLASSES
for a given feature cannot change an argmax over classes. That kills LERF's and LangSplat's
canonical-negative trick as used in those papers:
    r_c = min_n sigma(10*(cos(f,t_c) - cos(f,t_n))) = sigma(10*(cos(f,t_c) - max_n cos(f,t_n)))
since sigma is monotone and max_n cos(f,t_n) does not depend on c. It shifts every class
equally. It works for LERF because LERF thresholds a SINGLE query at 0.5 (a detection
problem); we rank classes against each other. Confirmed against both codebases.

WHAT CAN WORK is anything that varies PER CLASS:
  templates      prompt ensembling over the standard CLIP ImageNet templates -- changes each
                 class vector by a different amount. Neither OpenGaussian, LangSplat nor LERF
                 does this; all three use raw names.
  neg_class      LangSplat's get_semantic_map: append the canonical negatives as EXTRA
                 CLASSES and reject anything that argmaxes onto one. Not an offset -- a
                 reject option, so it does change the outcome.
  centering      our existing sim - lambda*column_mean(sim), the per-class mean over cells.
  zscore         per-class standardisation of similarities across cells (mean AND variance).
  whiten         mean-subtract the text bank itself, removing the shared text-cone direction
                 that makes every class similar to everything.
  gap_shift      subtract the (image_mean - text_mean) offset from the image features: a
                 direct estimate of the CLIP modality gap, computed WITHOUT labels.
  prototype      upper reference: replace text vectors with class-mean image features, fitted
                 on half the cells. Not zero-shot, included to show the ceiling.
  showercurtain  OpenGaussian queries "showercurtain"; we query "shower curtain". Tests
                 whether that single string costs anything.
"""
import argparse
import json
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, remap_gt_labels,
                                       load_scannet_pointcept_gt, calculate_metrics,
                                       CLIP_MODEL, CLIP_PRETRAINED)
from point_cloud_query import assign_points_to_power_cells
from diagnose_scannet_miou import load_foam
from run_cluster_classify_eval import SCENES

# the standard CLIP ImageNet prompt set, trimmed to the ones meaningful for indoor scans
TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a photo of one {}.",
    "itap of a {}.", "a bad photo of a {}.", "a origami {}.",
    "a photo of the large {}.", "a photo of the small {}.",
    "a cropped photo of a {}.", "a close-up photo of a {}.",
    "a bright photo of a {}.", "a dark photo of a {}.",
    "a photo of a {} in a room.", "there is a {} in the scene.",
    "a rendering of a {}.", "a low resolution photo of a {}.",
]
NEGATIVES = ["object", "things", "stuff", "texture"]


def encode(texts, device, model=None, tok=None):
    if model is None:
        model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
        model.eval().to(device)
        tok = open_clip.get_tokenizer(CLIP_MODEL)
    with torch.no_grad():
        t = tok(texts).to(device)
        f = model.encode_text(t).float()
    return F.normalize(f, dim=-1), model, tok


def encode_ensemble(names, device, model, tok):
    """Prompt ensembling: mean of the normalized embeddings over templates, renormalized."""
    out = []
    for n in names:
        prompts = [t.format(n) for t in TEMPLATES]
        with torch.no_grad():
            e = model.encode_text(tok(prompts).to(device)).float()
        e = F.normalize(e, dim=-1).mean(0)
        out.append(e)
    return F.normalize(torch.stack(out), dim=-1)


def score(pred_cls, vi_np, centers_n, assigned, owned, gt_t, n_classes):
    pc = np.zeros(centers_n, dtype=np.int64)
    pc[vi_np] = pred_cls
    pred = np.zeros(len(gt_t), dtype=np.int64)
    pred[owned] = pc[assigned[owned]] + 1
    _, miou, _, macc = calculate_metrics(torch.from_numpy(gt_t).long(),
                                         torch.from_numpy(pred).long(), n_classes)
    return float(miou), float(macc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scene0347_00,scene0070_00,scene0140_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
    p.add_argument("--class-set", default="opengaussian19")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    device = "cuda"
    model = tok = None
    per_scene = {}
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
        vi = torch.where(torch.from_numpy(vm).to(device))[0]
        unit = torch.zeros_like(feats)
        unit[vi] = F.normalize(feats[vi], dim=-1)
        X = unit[vi]
        vi_np = vi.cpu().numpy()

        n2i = {n: i for i, n in enumerate(all_names)}
        present = set(np.unique(raw_labels).tolist())
        kept = [(n2i[n], n) for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in present]
        tids, tnames = [i for i, _ in kept], [n for _, n in kept]
        gt_t = remap_gt_labels(raw_labels, tids)
        n_classes = len(tids) + 1
        assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=vm, k=64)
        owned = assigned >= 0

        text, model, tok = encode(tnames, device, model, tok)
        neg, _, _ = encode(NEGATIVES, device, model, tok)
        sim = X @ text.T

        rows = {}
        rows["baseline (raw names, argmax)"] = score(sim.argmax(-1).cpu().numpy(), vi_np,
                                                     centers.shape[0], assigned, owned,
                                                     gt_t, n_classes)
        # --- per-class interventions (these CAN change the argmax) ---
        rows["centering (lam=%.2f)" % a.lam] = score(
            (sim - a.lam * sim.mean(0, keepdim=True)).argmax(-1).cpu().numpy(),
            vi_np, centers.shape[0], assigned, owned, gt_t, n_classes)
        z = (sim - sim.mean(0, keepdim=True)) / sim.std(0, keepdim=True).clamp_min(1e-6)
        rows["per-class z-score"] = score(z.argmax(-1).cpu().numpy(), vi_np, centers.shape[0],
                                          assigned, owned, gt_t, n_classes)
        tw = F.normalize(text - text.mean(0, keepdim=True), dim=-1)
        rows["whitened text bank"] = score((X @ tw.T).argmax(-1).cpu().numpy(), vi_np,
                                           centers.shape[0], assigned, owned, gt_t, n_classes)
        gap = X.mean(0, keepdim=True) - text.mean(0, keepdim=True)
        Xg = F.normalize(X - gap, dim=-1)
        rows["modality-gap shift"] = score((Xg @ text.T).argmax(-1).cpu().numpy(), vi_np,
                                           centers.shape[0], assigned, owned, gt_t, n_classes)
        te = encode_ensemble(tnames, device, model, tok)
        rows["prompt ensemble (16 templates)"] = score((X @ te.T).argmax(-1).cpu().numpy(),
                                                       vi_np, centers.shape[0], assigned,
                                                       owned, gt_t, n_classes)
        rows["ensemble + centering"] = score(
            ((X @ te.T) - a.lam * (X @ te.T).mean(0, keepdim=True)).argmax(-1).cpu().numpy(),
            vi_np, centers.shape[0], assigned, owned, gt_t, n_classes)
        # LangSplat get_semantic_map: negatives as EXTRA CLASSES with a reject option
        both = torch.cat([text, neg], 0)
        am = (X @ both.T).argmax(-1)
        rej = am >= len(tids)
        am = am.clone(); am[rej] = 0                     # rejected -> class 0, then masked
        pc = np.zeros(centers.shape[0], dtype=np.int64)
        pc[vi_np] = am.cpu().numpy()
        pred = np.zeros(len(gt_t), dtype=np.int64)
        pred[owned] = pc[assigned[owned]] + 1
        pred[np.isin(assigned, vi_np[rej.cpu().numpy()])] = 0
        _, mi, _, ma = calculate_metrics(torch.from_numpy(gt_t).long(),
                                         torch.from_numpy(pred).long(), n_classes)
        rows["negatives as extra classes"] = (float(mi), float(ma))

        per_scene[scene] = rows
        print(f"\n=== {scene} ({len(tids)} classes) ===")
        base = rows["baseline (raw names, argmax)"][0]
        for k, (mi_, ma_) in rows.items():
            print(f"  {k:<34} mIoU={mi_*100:6.2f} mAcc={ma_*100:6.2f}  ({(mi_-base)*100:+.2f})")

    print("\n=== mean over scenes (delta vs baseline) ===")
    keys = list(next(iter(per_scene.values())).keys())
    base_k = keys[0]
    for k in keys:
        d = np.mean([per_scene[s][k][0] - per_scene[s][base_k][0] for s in per_scene]) * 100
        m = np.mean([per_scene[s][k][0] for s in per_scene]) * 100
        print(f"  {k:<34} mIoU={m:6.2f}  ({d:+.2f})")
    if a.output:
        json.dump(per_scene, open(a.output, "w"), indent=2)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
