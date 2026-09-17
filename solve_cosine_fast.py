"""Cosine-composite lifting, exact and without any (rays, F) array -- full view budget.

Same objective as `solve_cosine_composite.py`:

    min_{||u_j||=1}  J(U) = sum_i ( 1 - < r_i/||r_i|| , b_{s(i)} > ),   r_i = sum_j A_ij u_j

but reorganised so nothing of size (rays, F) is ever formed. The naive version materialises three
such arrays per view per iteration (r, b = tab[seg], dJ/dr) -- ~7.5 GB of transient at
1296x968x512 -- which is why it had to be run on 12 views, which in turn made the comparison
against an all-view baseline meaningless (A28.6). Removing the transient removes the reason to
subsample views at all.

MEMORY, HONESTLY. Removing the (rays, F) arrays does NOT make this free: the pair terms below
touch (pairs, F), and for foam pairs ~ 1.5 x rays, so one large transient replaced another. The
measured gain is ~2x per view, not the two orders of magnitude a first estimate suggested. Both
the pair products and the cached per-view indices are therefore chunked and stored as int32 --
without that, a single view of scene0070 asks for 12.8 GB and 279 cached views of scene0000 ask
for ~15 GB of pair lists alone.

THE TWO COLLAPSES. With n_i = ||r_i|| and c_i = <r_i/n_i, b_{s(i)}>,

    dJ/du_j = - sum_i (A_ij/n_i) b_{s(i)}  +  sum_i (c_i/n_i^2) A_ij r_i

Term 1: b is piecewise constant over SAM segments, so group rays by segment --
    sum_i (A_ij/n_i) b_{s(i)} = (M B)_j ,   M_{js} = sum_{i: s(i)=s} A_ij / n_i
with M only (P, S), S ~ 14-142. The per-pixel expansion of `tab` is ~1e5x redundant and is never
built.

Term 2: substitute r_i = sum_k A_ik u_k --
    sum_i (c_i/n_i^2) A_ij r_i = (G_w U)_j ,   (G_w)_jk = sum_i (c_i/n_i^2) A_ij A_ik
a RE-WEIGHTED GRAM on the same co-visibility sparsity pattern, so it is one sparse matvec.

The two scalars need no (rays, F) array either:
    n_i^2 = sum_j A_ij^2 + sum_{j != k} A_ij A_ik <u_j, u_k>
    <r_i, b_s> = sum_j A_ij (U B^T)_{js}
<u_j,u_k> is required only on co-visible pairs -- exactly the pairs this script enumerates -- so
it costs one |pairs| pass, and U B^T is a small (P, S) matmul.

Exact: same objective, same gradient, same fixed point. `--check` verifies the gradient against
finite differences AND the fast objective against a direct dense evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import configargparse

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
from configs import Params, add_group
from data_loader import DataHandler
from diagnose_holes import SCENES


def build_pairs(row, col, val, dev, pair_budget=1 << 23):
    """Per-ray co-hit pairs (a, b, w=A_ia*A_ib, ray) for the ||r_i||^2 cross terms.

    Rays are grouped by hit count so the k x k upper triangle is built once per group rather than
    padding every ray to the maximum -- the same grouping `build_covis_graph.py` uses, and for the
    same reason: mean k is ~2.3 but the max is far larger.
    """
    order = torch.argsort(row, stable=True)
    row, col, val = row[order], col[order], val[order]
    R = int(row.max().item()) + 1 if row.numel() else 0
    cnt = torch.bincount(row, minlength=R)
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, 0)[:-1]
    pa, pb, pw, pr = [], [], [], []
    for k in torch.unique(cnt):
        ki = int(k)
        if ki < 2:
            continue
        rows = (cnt == k).nonzero(as_tuple=True)[0]
        ii, jj = torch.triu_indices(ki, ki, offset=1, device=dev)
        npair = ii.numel()
        # CHUNKED BY PAIR COUNT, not just grouped by k. A group with many rows at a large k
        # builds an (n_rows, k*(k-1)/2) expansion in one go -- at k = 64 that is 2016 pairs per
        # ray, and it was this, not the (pairs, F) products, that asked for 8-12 GB on the
        # high-hit scenes. Same guard `build_covis_graph.py` uses, for the same reason.
        step = max(1, pair_budget // max(npair, 1))
        ar = torch.arange(ki, device=dev)
        for s0 in range(0, rows.numel(), step):
            rr = rows[s0:s0 + step]
            idx = starts[rr][:, None] + ar[None, :]
            c = col[idx]
            v = val[idx]
            pa.append(c[:, ii].reshape(-1))
            pb.append(c[:, jj].reshape(-1))
            pw.append((v[:, ii] * v[:, jj]).reshape(-1))
            pr.append(rr.repeat_interleave(npair))
            del idx, c, v
    if not pa:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return row, col, val, z, z, torch.zeros(0, device=dev), z
    return (row, col, val, torch.cat(pa), torch.cat(pb), torch.cat(pw), torch.cat(pr))


def load_view_B(feat_dir, stem, H, W, dev):
    f = np.load(os.path.join(feat_dir, f"{stem}_f.npy"))
    s = np.load(os.path.join(feat_dir, f"{stem}_s.npy"))
    if s.ndim == 3:
        s = s[0]
    tab = F.normalize(torch.from_numpy(np.ascontiguousarray(f)).float().to(dev), dim=-1)
    seg = torch.from_numpy(np.ascontiguousarray(s)).long().to(dev)
    if tuple(seg.shape) != (H, W):
        seg = F.interpolate(seg[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    return seg.reshape(-1), tab


class View:
    __slots__ = ("row", "col", "val", "pa", "pb", "pw", "pr", "seg", "tab", "nray", "diag")

    def __init__(self, row, col, val, seg, tab, nray, dev):
        r, c, v, pa, pb, pw, pr = build_pairs(row, col, val, dev)
        self.row, self.col, self.val = r.to(torch.int32), c.to(torch.int32), v
        self.pa, self.pb = pa.to(torch.int32), pb.to(torch.int32)
        self.pw, self.pr = pw, pr.to(torch.int32)
        self.seg, self.tab, self.nray = seg, tab, nray
        self.diag = torch.zeros(nray, device=dev).index_add_(0, r, v * v)   # sum_j A_ij^2


def obj_grad(U, views, P, Fd, dev, want_grad=True):
    """J and dJ/dU with no (rays, F) intermediate anywhere."""
    tot = 0.0
    G = torch.zeros(P, Fd, device=dev) if want_grad else None
    PC = max(1, int(4.0e8 // (4 * Fd)))                       # pair chunk: bounds (pairs, F)
    for vw in views:
        rw, cl = vw.row.long(), vw.col.long()
        # <r_i, b_s(i)> via the small (P,S) projection, never expanding tab per pixel
        Cu = U @ vw.tab.T                                    # (P, S)
        sflat = vw.seg.clamp_min(0)
        num = torch.zeros(vw.nray, device=dev)
        num.index_add_(0, rw, vw.val * Cu[cl, sflat[rw]])
        # ||r_i||^2 = sum_j A_ij^2 + 2 sum_{j<k} A_ij A_ik <u_j,u_k>
        n2 = vw.diag.clone()
        for s0 in range(0, vw.pa.numel(), PC):
            e_ = slice(s0, s0 + PC)
            pa, pb = vw.pa[e_].long(), vw.pb[e_].long()
            e = (U[pa] * U[pb]).sum(-1)                      # <u_j,u_k> on co-hit pairs only
            n2.index_add_(0, vw.pr[e_].long(), 2.0 * vw.pw[e_] * e)
            del e, pa, pb
        n2 = n2.clamp_min(1e-20)
        n = n2.sqrt()
        live = (vw.seg >= 0) & (vw.diag > 0)
        cos = (num / n).clamp(-1, 1)
        tot += float((1.0 - cos)[live].sum())
        if want_grad:
            lw = live.float()
            # term 1: -(M B), M_{js} = sum_{i: s(i)=s} A_ij / n_i
            M = torch.zeros(P, vw.tab.shape[0], device=dev)
            M.index_put_((cl, sflat[rw]), vw.val * (lw / n)[rw], accumulate=True)
            G -= M @ vw.tab
            # term 2: +(G_w U), G_w = A^T diag(cos/n^2) A on the co-hit pattern
            w = (cos * lw) / n2
            G.index_add_(0, cl, (vw.val * vw.val * w[rw]).unsqueeze(-1) * U[cl])
            for s0 in range(0, vw.pa.numel(), PC):
                e_ = slice(s0, s0 + PC)
                pa, pb = vw.pa[e_].long(), vw.pb[e_].long()
                wp = vw.pw[e_] * w[vw.pr[e_].long()]
                G.index_add_(0, pa, wp.unsqueeze(-1) * U[pb])
                G.index_add_(0, pb, wp.unsqueeze(-1) * U[pa])
                del wp, pa, pb
            del M
        del Cu, num, n2, n, cos, rw, cl
    return tot, G


def run(scene, recon, n_views, iters, cap, feat_dir_name, dev="cuda"):
    import warp as wp
    from powerfoam.feature_operator import export_operator_for_views
    from powerfoam.scene import PowerfoamScene
    wp.init()
    ck = f"output/scannet_{scene}_{recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ck}/model.pt")
    P = model.points.shape[0]

    names = sorted(os.listdir(f"data/scannet/{scene}_colmap/images"))
    feat_dir = f"data/scannet/{scene}_colmap/{feat_dir_name}"
    nv = len(dh.cameras) if n_views <= 0 else min(n_views, len(dh.cameras))
    sel = (list(range(len(dh.cameras))) if n_views <= 0
           else np.linspace(0, len(dh.cameras) - 1, nv).astype(int).tolist())

    views, Fd = [], None
    for vi in sel:
        stem = os.path.splitext(names[vi])[0]
        if not os.path.exists(os.path.join(feat_dir, f"{stem}_f.npy")):
            continue
        cam = dh.cameras[vi]
        H, W = int(cam.height), int(cam.width)
        op = export_operator_for_views(model, [cam], [vi], max_hits_per_pixel=cap,
                                       max_intersections=4096)
        row, col, val = (op.row_indices.to(torch.int64), op.col_indices.to(torch.int64),
                         op.values.float())
        del op
        seg, tab = load_view_B(feat_dir, stem, H, W, dev)
        views.append(View(row, col, val, seg, tab, H * W, dev))
        Fd = tab.shape[1]
    if not views:
        raise RuntimeError("no views with features")

    # matched baseline: the closed form over EXACTLY these views (A28.6)
    AtB = torch.zeros(P, Fd, device=dev)
    Dv = torch.zeros(P, device=dev)
    for vw in views:
        sflat = vw.seg.clamp_min(0)
        rw, cl = vw.row.long(), vw.col.long()
        ok = (vw.seg >= 0)[rw]
        AtB.index_add_(0, cl[ok], vw.val[ok].unsqueeze(-1) * vw.tab[sflat[rw[ok]]])
        Dv.index_add_(0, cl[ok], vw.val[ok])
        del rw, cl, ok
    live = Dv > 0
    W0 = torch.zeros(P, Fd, device=dev)
    W0[live] = AtB[live] / Dv[live].unsqueeze(-1)
    W0 = F.normalize(W0, dim=-1)
    W0[~live] = 0.0
    torch.save({"primitive_features": W0.cpu().half(), "valid_mask": live.cpu()},
               f"artifacts/scannet/{scene}/solved_wfull_{recon}_ogl3.pt")

    U = W0.clone()
    j0, G = obj_grad(U, views, P, Fd, dev)
    eta = 1.0 / G.norm(dim=-1).max().clamp_min(1e-30)
    best, jb, bit = U.clone(), j0, 0
    for it in range(iters):
        _, G = obj_grad(U, views, P, Fd, dev)
        Un = F.normalize(U - eta * G, dim=-1)
        Un[~live] = 0.0
        j, _ = obj_grad(Un, views, P, Fd, dev, want_grad=False)
        if j < jb:
            jb, best, bit, U = j, Un.clone(), it + 1, Un
        else:
            eta *= 0.5
            U = Un
    return best, dict(scene=scene, recon=recon, P=int(P), views=len(views),
                      obj_init=j0, obj_best=jb, best_iter=bit,
                      obj_drop=float(1.0 - jb / max(j0, 1e-30)), iters=iters)


def check(dev="cpu"):
    """Fast path vs a dense reference: same objective, and gradient vs AUTOGRAD.

    Autograd, not finite differences. A float32 central difference with eps=1e-5 on an objective
    of magnitude ~70 loses most of its significant digits to cancellation and reported a 1.9e-01
    "error" against a gradient that is in fact correct to 2e-07 -- a broken test that nearly
    discarded a correct implementation. Autograd has no step size to get wrong.
    """
    g = torch.Generator().manual_seed(0)
    nray, P, Fd, S = 60, 14, 5, 4
    seg = torch.randint(0, S, (nray,), generator=g)
    tab = F.normalize(torch.randn(S, Fd, generator=g), dim=-1)
    rows, cols, vals = [], [], []
    for i in range(nray):
        k = int(torch.randint(1, 4, (1,), generator=g))
        c = torch.randperm(P, generator=g)[:k]
        v = torch.rand(k, generator=g) + 0.05
        v = v / v.sum()
        rows.append(torch.full((k,), i)); cols.append(c); vals.append(v)
    row, col, val = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    vw = View(row, col, val, seg, tab, nray, dev)
    U = F.normalize(torch.randn(P, Fd, generator=g), dim=-1)

    A = torch.zeros(nray, P); A[row, col] = val

    def Jref(Um):
        r = A @ Um
        return float((1.0 - (F.normalize(r, dim=-1) * tab[seg]).sum(-1)).sum())

    jf, Gf = obj_grad(U, [vw], P, Fd, dev)
    jr = Jref(U)
    print(f"[check] objective fast {jf:.10f} vs dense {jr:.10f}  |diff| {abs(jf - jr):.2e}")
    Ug = U.clone().requires_grad_(True)
    rr = A @ Ug
    Jt = (1.0 - (F.normalize(rr, dim=-1) * tab[seg]).sum(-1)).sum()
    Jt.backward()
    rel = float((Gf - Ug.grad).abs().max() / Ug.grad.abs().max().clamp_min(1e-30))
    print(f"[check] gradient vs autograd, max relative error {rel:.2e}  -> "
          f"{'OK' if rel < 1e-4 else 'FAIL'}")
    return abs(jf - jr) < 1e-6 and rel < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--recons", default="truefrozen")
    ap.add_argument("--views", type=int, default=0, help="0 = ALL views")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--feature-folder", default="openclip_features_sam_l3")
    ap.add_argument("--tag", default="cosfast")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="artifacts/scannet/cosine_fast.json")
    a = ap.parse_args()
    if a.check and not check():
        raise SystemExit("check failed -- not running")
    rows = []
    for rec in a.recons.split(","):
        for sc in a.scenes.split(","):
            try:
                U, info = run(sc, rec, a.views, a.iters, a.cap, a.feature_folder)
            except Exception as e:
                print(f"[{rec}/{sc}] SKIP {type(e).__name__}: {e}", flush=True)
                continue
            torch.save({"primitive_features": U.cpu().half(),
                        "valid_mask": torch.load(
                            f"artifacts/scannet/{sc}/solved_wfull_{rec}_ogl3.pt",
                            map_location="cpu", weights_only=True)["valid_mask"]},
                       f"artifacts/scannet/{sc}/solved_{a.tag}_{rec}_ogl3.pt")
            rows.append(info)
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"[{rec}/{sc}] views {info['views']} obj {info['obj_init']:.4e} -> "
                  f"{info['obj_best']:.4e} ({info['obj_drop']:+.2%}) best@{info['best_iter']}",
                  flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
