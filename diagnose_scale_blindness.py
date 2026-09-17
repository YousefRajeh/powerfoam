"""Why did a catastrophically wrong solver score a near-correct mIoU?

The block solver shipped without diagonal lumping had L = +2.45e8 against the closed form's
-4.95e7 -- four orders of magnitude wrong on the objective -- yet scored 0.382 mIoU against the
closed form's 0.390.

HYPOTHESIS. The failure mode is per-primitive INFLATION, and the evaluation is invariant to it.
For a singleton block the bad solve is x_j = A^T B_j / S_jj where the correct one is
A^T B_j / D_jj, so x_bad_j = (D_jj / S_jj) x_good_j -- a positive scalar multiple, i.e. the SAME
DIRECTION. And run_cluster_classify_eval.py normalises (`F.normalize(feats, dim=-1)`) before
spherical k-means and before the cosine similarity against CLIP text embeddings, so nothing
downstream can see a per-primitive scale.

If true, the mIoU pipeline is structurally blind to a large part of what the solver controls, and
"L went down but mIoU went down too" is not two objectives disagreeing -- it is one objective being
partly invisible to the other.

This measures the per-primitive cosine between the two solutions, and the norm ratio, to say how
much of the difference is pure scale.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F

import run_solver_variants as rsv

SCENE = "scene0347_00"
CAP = 4
dev = "cuda"

P, keys, vals, Atb, sup = rsv.load_cache(SCENE, dev)
S = rsv.build_S(keys, vals, P, dev)
groups = rsv.surface_blocks(SCENE, keys, vals, P, CAP, dev)

x_good = rsv.solve_blockdiag(S, Atb, sup, groups, dev)

# --- rebuild the BAD solve: blockdiag(S) with the off-block mass discarded ---------------------
j, l, v, off = S
bi = torch.cat([j, l[off]]); bj = torch.cat([l, j[off]]); bv = torch.cat([v, v[off]])
same = groups[bi] == groups[bj]
bi, bj, bv = bi[same], bj[same], bv[same]
order = torch.argsort(groups)
gsorted = groups[order]
sizes = torch.bincount(groups)
starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                                 torch.bincount(gsorted)[:-1]]), 0)
x_bad = torch.zeros_like(Atb)
for sz in range(1, int(sizes.max()) + 1):
    gsel = (sizes == sz).nonzero(as_tuple=True)[0]
    if not gsel.numel():
        continue
    G = gsel.numel()
    rows = order[starts[gsel].unsqueeze(1) + torch.arange(sz, device=dev).unsqueeze(0)]
    flat = rows.reshape(-1)
    loc = torch.full((P,), -1, dtype=torch.long, device=dev)
    loc[flat] = torch.arange(G * sz, device=dev) % sz
    gof = torch.full((P,), -1, dtype=torch.long, device=dev)
    gof[flat] = torch.arange(G, device=dev).repeat_interleave(sz)
    M = torch.zeros((G, sz, sz), device=dev)
    sel = (gof[bi] >= 0) & (gof[bi] == gof[bj])
    M[gof[bi][sel], loc[bi][sel], loc[bj][sel]] = bv[sel]      # NO diagonal lumping = the bug
    M += torch.eye(sz, device=dev) * 1e-12
    x_bad[flat] = torch.linalg.solve(M, Atb[flat].reshape(G, sz, -1)).reshape(G * sz, -1)
    del rows, flat, loc, gof, M
    torch.cuda.empty_cache()

m = (sup > 0) & torch.isfinite(x_bad).all(1) & (x_good.norm(dim=1) > 0) & (x_bad.norm(dim=1) > 0)
cos = F.cosine_similarity(x_bad[m], x_good[m], dim=1)
ratio = x_bad[m].norm(dim=1) / x_good[m].norm(dim=1)
gsz = sizes[groups][m]

print(f"\ncompared on {int(m.sum()):,} primitives of {P:,}")
print(f"  cosine(x_bad, x_good):  mean {cos.mean():.6f}  median {cos.median():.6f}  "
      f"p5 {cos.quantile(0.05):.6f}")
print(f"  frac cos > 0.99      :  {(cos > 0.99).float().mean():.4f}")
print(f"  frac cos > 0.999     :  {(cos > 0.999).float().mean():.4f}")
print(f"  norm ratio bad/good  :  median {ratio.median():.2f}  p95 {ratio.quantile(0.95):.2f}  "
      f"max {ratio.max():.1f}")
for sz in sorted(set(gsz.tolist())):
    sel = gsz == sz
    print(f"    block size {sz}: n={int(sel.sum()):>8,}  cos median {cos[sel].median():.6f}  "
          f"norm ratio median {ratio[sel].median():.2f}")
print("\nreading: cosine ~1 with a large norm ratio means the error is almost pure per-primitive")
print("SCALE, which F.normalize() in the evaluation removes entirely.")
