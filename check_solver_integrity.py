"""Integrity gate for the solver variants: does every file on disk satisfy the theory?

Two sweep processes briefly shared a log, so some `solved_*.pt` files may have been written by a
different code revision, or raced. Rather than reason about which, this re-derives the invariants
each file must satisfy and fails loudly on any scene that violates one:

  1. richardson_k1 == A^T B / support             (k=1 IS the closed form, SFS Eq. 6)
  2. L(k1) > L(k2) > L(k10)                       (Richardson is strictly monotone in L)
  3. L(surfblock) < L(k1)                         (block lumping is a better preconditioner than D)
  4. no NaN/Inf anywhere

L is reported as x^T S x - 2<x, A^T B>, i.e. up to the constant ||B||^2, which is identical across
variants of the same scene.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

VARIANTS = ["richardson_k1", "richardson_k2", "richardson_k10", "surfblock_c4"]
SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bad = []
    print(f"{'scene':<15}{'L(k1)':>16}{'L(k2)':>16}{'L(k10)':>16}{'L(blk)':>16}  verdict")
    for sc in SCENES:
        cp = [q for q in sorted(glob.glob(f"artifacts/scannet/{sc}/gram_cache_K6_l3_*.pt"))
              if "bgdrop" not in q]
        if not cp:
            print(f"{sc:<15} no cache"); continue
        c = torch.load(cp[0], map_location="cpu", weights_only=False)
        P = int(c["P"])
        keys, vals = c["S_keys"], c["S_vals"].float()
        Atb = c["Atb"].float().to(dev)
        sup = c["support"].float().to(dev)
        for k in list(c):
            c[k] = None
        del c
        j, l = (keys // P).to(dev), (keys % P).to(dev)
        v = vals.to(dev)
        off = j != l
        del keys, vals

        def Sx(x, ec=1 << 23, cc=64):
            out = torch.zeros_like(x)
            for a in range(0, x.shape[1], cc):
                s_ = slice(a, min(a + cc, x.shape[1]))
                xc, oc = x[:, s_], out[:, s_]
                for b in range(0, j.numel(), ec):
                    e = min(b + ec, j.numel())
                    js, ls, vs, om = j[b:e], l[b:e], v[b:e], off[b:e]
                    oc.index_add_(0, js, vs.unsqueeze(1) * xc[ls])
                    oc.index_add_(0, ls[om], vs[om].unsqueeze(1) * xc[js[om]])
                out[:, s_] = oc
            return out

        Ls, issues = {}, []
        for name in VARIANTS:
            f = f"artifacts/scannet/{sc}/solved_{name}.pt"
            if not os.path.exists(f):
                issues.append(f"missing {name}"); continue
            x = torch.load(f, map_location=dev, weights_only=True)["primitive_features"].float()
            if not torch.isfinite(x).all():
                issues.append(f"{name} non-finite")
            Ls[name] = float((x * Sx(x)).sum() - 2.0 * (x * Atb).sum())
            if name == "richardson_k1":
                ref = torch.zeros_like(Atb)
                m = sup > 0
                ref[m] = Atb[m] / sup[m].unsqueeze(1)
                rel = float((x - ref).norm() / ref.norm().clamp_min(1e-30))
                if rel > 1e-4:
                    issues.append(f"k1 != A^T B/support (rel {rel:.1e})")
            del x
            torch.cuda.empty_cache()

        if all(k in Ls for k in VARIANTS):
            if not Ls["richardson_k1"] > Ls["richardson_k2"] > Ls["richardson_k10"]:
                issues.append("L not monotone in k")
            if not Ls["surfblock_c4"] < Ls["richardson_k1"]:
                issues.append("surfblock worse than k1")
        verdict = "OK" if not issues else "FAIL: " + "; ".join(issues)
        if issues:
            bad.append(sc)
        print(f"{sc:<15}" + "".join(f"{Ls.get(n, float('nan')):>16.6g}" for n in VARIANTS)
              + f"  {verdict}", flush=True)
        del j, l, v, off, Atb, sup
        torch.cuda.empty_cache()

    print(f"\n{len(SCENES) - len(bad)}/{len(SCENES)} scenes pass"
          + (f"; FAILED: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
