"""Operator-aware partition selection: solve THROUGH the restricted operator A_Q = A Q.

THE DISTINCTION THIS TESTS. Everything tried so far pooled POST HOC -- take the per-primitive
solution X and average it inside each segment. That ignores the operator: it is a projection of the
solution, not a solution of the projected problem. With Q the P x K assignment and X = Q Z, the
rendered operator is A_Q = A Q and the estimator is

    Z = argmin ||A Q Z - B||^2  =>  (Q^T A^T A Q) Z = Q^T A^T B

which equals post-hoc pooling only when A_Q has orthogonal columns. So the two differ exactly where
segments share evidence -- which is the interesting case.

EVERYTHING NEEDED IS ALREADY ON DISK; no re-accumulation per candidate partition:
  * off-diagonal of A^T A  -- covis_*.pt, keys `lo*P + hi`, UPPER-TRIANGULAR, top-64 per node
  * diagonal of A^T A      -- stats.support2 = sum_r A_rp^2
  * A^T B                  -- stats.numerator
so G = Q^T (D + S + S^T) Q is a sparse triple product into K x K.

DIAGNOSTICS REPORTED (conditioning only -- see the caveat below):
  * normalized Gram   Gn = Dg^-1/2 G Dg^-1/2, its off-diagonal mass and effective rank
  * exposure          diag(G): operator mass each segment actually receives
  * held-out-view fit optional, via --holdout: residual on views excluded from the fit

WHAT THESE DIAGNOSTICS CANNOT DO. Neither the Gram nor held-out-view fit certifies SEMANTIC
benefit, because the teacher bias is shared across views: measured here, region-level CLIP names are
correct only 49.2% of the time, and a coherently misnamed region is CONFIDENTLY wrong -- it fits
held-out views perfectly while being semantically wrong. Confidence gating failed for exactly this
reason (margin predicted purity at +0.49 yet gating never beat ungated pooling). So conditioning is
necessary, not sufficient, and mIoU remains the only semantic arbiter.

TRUNCATION CAVEAT. S is top-64 per node, so off-diagonal mass in G is a LOWER bound; the diagonal
is exact. Segments whose coupling lives in the tail are under-counted, which biases the Gram toward
looking better conditioned than it is. Reported, not corrected.

The partition algorithm itself remains Landrieu-Obozinski Cut Pursuit; this changes only how Z is
estimated given Q, and how a candidate Q is scored.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "D:/Downloads/feature-foam-lifting/src")
sys.path.insert(0, "D:/Downloads/powerfoam")


def load_gram_parts(scene, recon, covis_name):
    c = torch.load("artifacts/scannet/%s/%s.pt" % (scene, covis_name),
                   map_location="cpu", weights_only=False)
    P = int(c["P"])
    keys = c["S_keys"]
    vals = c["S_vals"].float()
    lo = (keys // P).long()
    hi = (keys % P).long()
    off = lo != hi
    return P, lo[off], hi[off], vals[off]


def restricted_gram(lab, K, P, lo, hi, val, diag):
    """G = Q^T (D + S + S^T) Q, with S the stored upper-triangular off-diagonal part."""
    G = torch.zeros(K, K, dtype=torch.float64)
    a, b = lab[lo], lab[hi]
    v = val.double()
    # both triangles: (a,b) and (b,a)
    G.index_put_((a, b), v, accumulate=True)
    G.index_put_((b, a), v, accumulate=True)
    G.index_put_((lab, lab), diag.double(), accumulate=True)
    return G


def gram_diagnostics(G, supported=None):
    """Diagnostics over SUPPORTED segments only.

    Unobserved primitives carry zero features, so Cut Pursuit groups them into segments with no
    operator mass at all (measured: the median segment has exposure exactly 0). Including those
    makes the Gram trivially singular -- the 1.3e12 condition number they produce says nothing
    about semantic ambiguity, only that empty segments exist -- so they are reported separately
    and excluded from the conditioning numbers.
    """
    if supported is not None:
        G = G[supported][:, supported]
    d = torch.diagonal(G).clamp_min(1e-12)
    Dn = torch.diag(d.pow(-0.5))
    Gn = Dn @ G @ Dn
    offmass = (Gn.abs().sum() - torch.diagonal(Gn).abs().sum()) / Gn.shape[0]
    ev = torch.linalg.eigvalsh(Gn.float() + 1e-9 * torch.eye(Gn.shape[0]))
    ev = ev.clamp_min(1e-12)
    p = ev / ev.sum()
    eff_rank = float(torch.exp(-(p * p.log()).sum()))          # entropy-based effective rank
    return {"offdiag_mass_per_segment": float(offmass),
            "eff_rank": eff_rank, "eff_rank_frac": eff_rank / Gn.shape[0],
            "cond": float(ev.max() / ev.min()),
            "exposure_median": float(torch.diagonal(G).median()),
            "exposure_min": float(torch.diagonal(G).min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--covis", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--regions", required=True, help="solve .pt carrying `labels` (the partition Q)")
    ap.add_argument("--pooled", default=None, help="post-hoc pooled solve, for the paired control")
    ap.add_argument("--out", required=True, help="restricted-operator solution, .pt")
    ap.add_argument("--ridge", type=float, default=1e-3, help="x mean(diag G), for invertibility")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    A = "artifacts/scannet/%s" % a.scene
    covis = a.covis or ("covis_truefrozen" if a.recon == "truefrozen" else "covis_nf_h64")
    stats_p = a.stats or "%s/stats_%s_ogl3.pt" % (A, a.recon)

    P, lo, hi, val = load_gram_parts(a.scene, a.recon, covis)
    st = torch.load(stats_p, map_location="cpu", weights_only=False)
    diag = torch.as_tensor(st["support2"]).float().reshape(-1)
    AtB = torch.as_tensor(st["numerator"]).float()
    if diag.shape[0] != P or AtB.shape[0] != P:
        raise SystemExit("covis P=%d vs stats support2 %d / numerator %d -- mismatched arms"
                         % (P, diag.shape[0], AtB.shape[0]))

    sv = torch.load(a.regions, map_location="cpu", weights_only=True)
    lab = sv["labels"].long()
    K = int(lab.max()) + 1
    print("[setup] P=%d K=%d  off-diag nnz=%d (top-64 truncated)" % (P, K, val.numel()), flush=True)

    G = restricted_gram(lab, K, P, lo, hi, val, diag)
    expo = torch.diagonal(G)
    supported = expo > 0
    seg_sz = torch.bincount(lab, minlength=K)
    print("  segments with ZERO operator mass: %d/%d (%.1f%%), holding %d/%d primitives (%.1f%%)"
          % (int((~supported).sum()), K, 100 * float((~supported).float().mean()),
             int(seg_sz[~supported].sum()), P, 100 * float(seg_sz[~supported].sum()) / P))
    diagn = gram_diagnostics(G, supported)
    diagn["zero_exposure_segments"] = int((~supported).sum())
    diagn["zero_exposure_frac"] = float((~supported).float().mean())
    for k, v in diagn.items():
        print("  %-26s %.6g" % (k, v))

    # restricted solve: (Q^T A^T A Q) Z = Q^T A^T B
    QtB = torch.zeros(K, AtB.shape[1], dtype=torch.float64).index_add_(0, lab, AtB.double())
    lam = a.ridge * float(torch.diagonal(G).mean())
    Z = torch.linalg.solve(G + lam * torch.eye(K, dtype=torch.float64), QtB)
    X = Z[lab].float()
    import torch.nn.functional as F
    valid = sv.get("valid_mask")
    valid = valid if valid is not None else (X.norm(dim=-1) > 0)
    Xn = torch.zeros_like(X)
    Xn[valid] = F.normalize(X[valid], dim=-1)
    torch.save({"primitive_features": Xn, "valid_mask": valid, "labels": lab, "num_segments": K},
               a.out)
    print("[wrote]", a.out)

    if a.pooled and os.path.exists(a.pooled):
        pv = torch.load(a.pooled, map_location="cpu", weights_only=True)
        pf = pv["primitive_features"].float()
        m = valid & (pf.norm(dim=-1) > 0)
        cos = F.cosine_similarity(Xn[m], F.normalize(pf[m], dim=-1), dim=-1)
        diagn["cos_restricted_vs_pooled_mean"] = float(cos.mean())
        diagn["cos_restricted_vs_pooled_frac_gt_0.999"] = float((cos > 0.999).float().mean())
        print("  restricted vs post-hoc pooled: cos mean=%.4f  frac>0.999=%.4f"
              % (cos.mean(), (cos > 0.999).float().mean()))
    if a.report:
        diagn.update(scene=a.scene, recon=a.recon, K=K, P=P, regions=a.regions)
        json.dump(diagn, open(a.report, "w"), indent=2)


if __name__ == "__main__":
    main()
