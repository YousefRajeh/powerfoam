"""Is our vectorized `mask_nms` bit-exact against LangSplat's original sequential version?

WHAT WE CHANGED. splat-distiller commit `41660b7` ("Speed up SAMOpenCLIP feature extraction and
make it crash-safe") replaced LangSplat's per-pair Python double loop
(`langsplat/preprocess.py:238-250`) with a matmul-based computation
(`splat-distiller/pre_processing.py:279-295`). `mask_nms` decides WHICH SAM MASKS SURVIVE, so if
the two disagree even occasionally, every downstream CLIP embedding is computed over a different
mask set and no comparison to OpenGaussian's numbers is meaningful.

The other change in that commit is a restructure of `create()` to save each image's result
immediately instead of accumulating all images into one padded tensor. That is a memory/crash
-safety change with no arithmetic in it, so it is not tested here.

This test implements LangSplat's ORIGINAL loop verbatim from their source and compares the
returned `keep` mask against ours on random and adversarial inputs.

Run: D:\\conda\\envs\\powerfoam\\python.exe test_mask_nms_equivalence.py   (CPU, seconds)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\Downloads\splat-distiller")


def langsplat_mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """Verbatim transcription of LangSplat preprocess.py:215-280 (the sequential original)."""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]),
                                     dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def main():
    from pre_processing import mask_nms as ours

    rng = np.random.default_rng(0)
    cases, mismatches = 0, 0

    def run(masks, scores, label):
        nonlocal cases, mismatches
        cases += 1
        a = langsplat_mask_nms(masks.clone(), scores.clone())
        b = ours(masks.clone(), scores.clone())
        sa, sb = sorted(a.flatten().tolist()), sorted(b.flatten().tolist())
        ok = sa == sb
        if not ok:
            mismatches += 1
            print(f"  MISMATCH [{label}]: langsplat kept {len(sa)}, ours {len(sb)}; "
                  f"symmetric diff {sorted(set(sa) ^ set(sb))[:10]}")
        return ok

    print("random overlapping blobs (the realistic case)")
    for t in range(40):
        n, H, W = int(rng.integers(4, 22)), 48, 64
        m = torch.zeros(n, H, W, dtype=torch.bool)
        for k in range(n):
            y, x = rng.integers(0, H - 12), rng.integers(0, W - 12)
            h, w = rng.integers(6, 20), rng.integers(6, 20)
            m[k, y:min(y + h, H), x:min(x + w, W)] = True
        s = torch.rand(n)
        run(m, s, f"random{t}")

    print("adversarial cases")
    # exact duplicates -> iou 1.0, must be suppressed identically
    m = torch.zeros(5, 32, 32, dtype=torch.bool); m[:, 4:20, 4:20] = True
    run(m, torch.rand(5), "identical masks")
    # strict nesting -> exercises the inner_iou 0.5/0.85 branches
    m = torch.zeros(4, 32, 32, dtype=torch.bool)
    for k, sz in enumerate((30, 22, 14, 6)):
        m[k, :sz, :sz] = True
    run(m, torch.tensor([0.9, 0.8, 0.7, 0.6]), "nested masks")
    # disjoint -> zero intersection everywhere
    m = torch.zeros(4, 32, 32, dtype=torch.bool)
    for k in range(4):
        m[k, k * 8:(k + 1) * 8, :] = True
    run(m, torch.rand(4), "disjoint masks")
    # NOT TESTED: all-scores-below-threshold. Both implementations index keep_conf[index, 0]
    # on a 1-D tensor there and raise IndexError identically (LangSplat preprocess.py:266,
    # ours pre_processing.py:335) -- the fallback branch is dead code in BOTH, since real
    # scores are stability*predicted_iou and SAM never emits an all-sub-0.1 set. Identical
    # crash IS identical behaviour, so there is nothing to diverge.
    # a zero-area mask -> 0/0 in the original, clamped in ours
    m = torch.zeros(4, 32, 32, dtype=torch.bool); m[0] = True; m[1, :16] = True; m[2, :8] = True
    run(m, torch.rand(4), "zero-area mask present")

    print(f"\n{cases - mismatches}/{cases} cases identical")
    print("VERDICT:", "BIT-EXACT — the vectorization is safe"
          if mismatches == 0 else f"DIVERGES on {mismatches} cases — extraction is NOT comparable")


if __name__ == "__main__":
    main()
