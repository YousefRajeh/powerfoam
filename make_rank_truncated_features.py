"""Write rank-R truncated copies of the solved feature files, so the compression claim (10e) can
be re-scored through the FULL headline pipeline (kmeans320 / position-aware pooling), not only the
bare per-cell argmax that run_rank_compress.py uses.

The stored file keeps the DECOMPRESSED (P, 512) field so every downstream consumer works unchanged;
the compression claim is about the (P, R) code + (512, R) dictionary that the file also carries, and
`bytes_ratio` records what that pair would cost. Storing the reconstruction rather than the code is
deliberate: the point of the experiment is whether the truncation loses accuracy, and rewriting the
consumers to take a code would confound that with an implementation change.

Basis is fitted on the OBSERVED cells only (valid_mask), matching run_rank_compress.py.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from determinism import enable_determinism
from run_cluster_classify_eval import SCENES
from run_rank_compress import right_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--stem", default="solved_geometric_median_nonfrozen",
                    help="filename stem before the suffix; e.g. "
                         "solved_geometric_median_truefrozen for the frozen arm")
    ap.add_argument("--src-suffix", default="_ogl3")
    ap.add_argument("--dst-suffix", default=None, help="default: <src>r<rank>")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    enable_determinism()
    dst_suffix = args.dst_suffix or f"{args.src_suffix}r{args.rank}"

    for scene in args.scenes:
        src = f"artifacts/scannet/{scene}/{args.stem}{args.src_suffix}.pt"
        dst = f"artifacts/scannet/{scene}/{args.stem}{dst_suffix}.pt"
        if not os.path.exists(src):
            print(f"[skip] {scene}")
            continue
        if os.path.exists(dst):
            # never silently replace an artifact another run may already have been scored against
            print(f"[exists, not overwriting] {dst}")
            continue
        d = torch.load(src, map_location="cpu", weights_only=True)
        phi = d["primitive_features"].float()
        vm = d["valid_mask"]
        V, sv = right_basis(phi[vm.bool()], args.device)
        Vr = V[:, :args.rank]
        code = phi.to(args.device) @ Vr
        out = dict(d)
        out["primitive_features"] = (code @ Vr.T).cpu()
        out["rank_truncation"] = {"rank": args.rank, "src": os.path.basename(src),
                                  "dictionary": Vr.cpu(),
                                  "bytes_ratio": (phi.shape[0] * args.rank + 512 * args.rank)
                                  / (phi.shape[0] * 512)}
        torch.save(out, dst)
        rel = float((out["primitive_features"].to(args.device) - phi.to(args.device)).norm()
                    / phi.to(args.device).norm())
        print(f"{scene}: R={args.rank} relerr {rel:.4f} -> {dst}")
        del code, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
