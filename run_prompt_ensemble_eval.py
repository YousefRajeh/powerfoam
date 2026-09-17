"""Can prompt ensembles separate a hyponym from its hypernym? Scored on the standard protocol.

THE TARGET. The confuser diagnosis found failing classes losing a TWO-WAY contest against a
near-collinear neighbour: `shower curtain` beats `curtain` only 8% of the time (text cos 0.836),
`cabinet` vs `counter` 46% (0.753), `wall` vs `picture` 32% (0.719). Linear surgery on the existing
prototypes did not convert -- Lowdin hurt, negative-projection came out inside noise -- because the
overlap is SPECIFIC rather than a shared nuisance direction. Prompt engineering attacks it upstream
instead: change the words, not the geometry of their embeddings.

⚠️ PROTOCOL. Our headline comparison against NormLift is the RAW class-name protocol with no
templates. Everything here is therefore an exploratory arm, not a drop-in improvement: adopting any
of it would require giving every baseline the same treatment, or the comparison is rigged. The raw
arm is re-scored in the same run so the delta is measured, never inherited.

THE VARIANTS, and what each is actually testing:

  raw        the class name alone. The protocol baseline.
  ens        the standard CLIP template ensemble (mean of normalised per-template embeddings).
             Tests whether generic prompt averaging -- known to help ImageNet zero-shot -- helps
             here. It should NOT fix hyponym/hypernym overlap, because every template is applied
             to both members of the pair equally.
  ctx        indoor-scene context templates. Tests whether grounding in "a room" separates classes
             whose confusion is contextual (wall/picture) rather than lexical.
  head       MODIFIER EMPHASIS, derived mechanically from the class name: a multi-word class is
             averaged with its modifier alone ("shower curtain" + "shower"), upweighting the token
             that distinguishes it from its head noun. This is the only variant aimed squarely at
             hypernym nesting, and it needs no hand-written per-class knowledge -- it reads the
             name. On ScanNet-20 exactly one class is multi-word, which caps its possible effect
             and is stated here rather than discovered in the results.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")

SPLIT = {"scene0347_00": "train", "scene0070_00": "train", "scene0140_00": "train",
         "scene0645_00": "val", "scene0590_00": "train", "scene0200_00": "train",
         "scene0097_00": "train", "scene0400_00": "train", "scene0062_00": "train",
         "scene0000_00": "train"}
POINTCEPT = os.environ.get("PF_POINTCEPT", r"D:\Downloads\scannet_pointcept")

ENS_TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a picture of a {}.",
    "there is a {} in the scene.", "a close-up photo of a {}.",
    "a cropped photo of a {}.", "a bright photo of a {}.", "a dark photo of a {}.",
]
CTX_TEMPLATES = [
    "a photo of a {} in a room.", "a {} in an indoor scene.",
    "an indoor photo of a {}.", "a {} inside a building.",
    "a photo of a {} in a house.",
]


def build_text(mode, names, device="cuda"):
    """(K, F) L2-normalised class prototypes under the chosen prompting scheme."""
    from evaluate_point_cloud_miou import embed_class_names

    def emb(phrases):
        return Fn.normalize(embed_class_names(phrases, device).float(), dim=-1)

    if mode == "raw":
        return emb(list(names))
    if mode in ("ens", "ctx"):
        tpl = ENS_TEMPLATES if mode == "ens" else CTX_TEMPLATES
        acc = None
        for t in tpl:
            e = emb([t.format(n) for n in names])
            acc = e if acc is None else acc + e
        return Fn.normalize(acc / len(tpl), dim=-1)
    if mode == "head":
        base = emb(list(names))
        multi = [(i, n) for i, n in enumerate(names) if len(n.split()) > 1]
        if not multi:
            return base
        mods = emb([" ".join(n.split()[:-1]) for _, n in multi])
        out = base.clone()
        idx = torch.tensor([i for i, _ in multi], device=base.device)
        out[idx] = Fn.normalize(base[idx] + mods, dim=-1)
        return out
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=list(SPLIT))
    ap.add_argument("--modes", nargs="*", default=["raw", "ens", "ctx", "head"])
    ap.add_argument("--class-sets", nargs="*", default=["opengaussian19", "opengaussian10"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="artifacts/prompt_ensemble_eval.json")
    a = ap.parse_args()

    from build_true_facet_graph import load_points_radii
    from determinism import enable_determinism
    from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, classify_primitives,
                                           load_scannet_pointcept_gt, remap_gt_labels)
    from point_cloud_query import assign_points_to_power_cells

    enable_determinism()

    # what the prompting does to the confuser cosines, before any image evidence
    full = list(OPENGAUSSIAN_CLASS_SETS["opengaussian19"])
    pairs = [("shower curtain", "curtain"), ("cabinet", "counter"), ("wall", "picture"),
             ("counter", "sink"), ("table", "floor")]
    print(f"{'pair':30s} " + " ".join(f"{m:>8s}" for m in a.modes))
    tmats = {m: build_text(m, full) for m in a.modes}
    for x, y in pairs:
        i, j = full.index(x), full.index(y)
        print(f"{x + ' / ' + y:30s} " +
              " ".join(f"{float(tmats[m][i] @ tmats[m][j]):8.3f}" for m in a.modes))
    for m in a.modes:
        g = (tmats[m] @ tmats[m].T).cpu().numpy()
        np.fill_diagonal(g, np.nan)
        print(f"  {m:6s} mean |offdiag| {np.nanmean(np.abs(g)):.4f}")
    print()

    out = {}
    for cs in a.class_sets:
        rows = []
        for scene in a.scenes:
            ck = f"output/scannet_{scene}_truefrozen"
            fp = f"artifacts/scannet/{scene}/solved_geometric_median_truefrozen_ogl3.pt"
            if not (os.path.isdir(ck) and os.path.exists(fp)):
                continue
            pts, raw, all_names = load_scannet_pointcept_gt(
                os.path.join(POINTCEPT, SPLIT[scene], scene), "segment20")
            n2i = {n: i for i, n in enumerate(all_names)}
            present = set(np.unique(raw).tolist())
            names = [n for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in present]
            if len(names) < 2:
                continue
            gt_lab = remap_gt_labels(raw, [n2i[n] for n in names])
            nc = len(names) + 1
            cc, rr = load_points_radii(ck)
            centers = np.asarray(cc, np.float64); radii = np.asarray(rr, np.float64)
            sd = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
            sigma = Fn.softplus(sd["density"].float(), beta=100).numpy().reshape(-1)
            alpha = 1.0 - np.exp(-np.maximum(sigma, 0.0) * 2.0 * radii)
            d = torch.load(fp, map_location="cpu", weights_only=True)
            feats = d["primitive_features"].float().cuda()
            vm = d["valid_mask"].numpy() if "valid_mask" in d else np.ones(len(centers), bool)
            owner = np.asarray(assign_points_to_power_cells(pts, centers, radii, valid=None, k=8))

            rec = {"scene": scene, "n_classes": len(names)}
            for m in a.modes:
                t = build_text(m, names)
                pcls = classify_primitives(feats, t).cpu().numpy() + 1
                pcls[~vm] = 0
                pcls[alpha < a.alpha] = 0
                pred = np.where(owner >= 0, pcls[np.clip(owner, 0, len(pcls) - 1)], 0)
                ious, accs = [], []
                for k in range(1, nc):
                    g = gt_lab == k
                    if not g.any():
                        continue
                    p = pred == k
                    i_ = float((g & p).sum()); u_ = float((g | p).sum())
                    ious.append(i_ / u_ if u_ else 0.0)
                    accs.append(i_ / float(g.sum()))
                rec[f"{m}_mIoU"] = float(np.mean(ious) * 100)
                rec[f"{m}_mAcc"] = float(np.mean(accs) * 100)
            rows.append(rec)
            print(f"  {cs:16s} {scene:14s} " +
                  " ".join(f"{m}={rec[f'{m}_mIoU']:6.2f}" for m in a.modes), flush=True)
        if rows:
            out[cs] = rows
            base = np.array([r["raw_mIoU"] for r in rows])
            print(f"\n{cs}: {len(rows)} scenes")
            print(f"{'mode':8s} {'mIoU':>7s} {'d':>7s} {'SE':>6s} {'win':>6s} {'mAcc':>7s}")
            for m in a.modes:
                v = np.array([r[f"{m}_mIoU"] for r in rows])
                ac = np.mean([r[f"{m}_mAcc"] for r in rows])
                dl = v - base
                se = dl.std(ddof=1) / np.sqrt(len(dl)) if m != "raw" else 0.0
                print(f"{m:8s} {v.mean():7.2f} {dl.mean():+7.2f} {se:6.2f} "
                      f"{int((dl > 0).sum()):>3d}/{len(dl):<2d} {ac:7.2f}")
            print()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
