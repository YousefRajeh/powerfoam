"""The exported 3DGS operator must satisfy the alpha-compositing invariant sum_i alpha_i T_i <= 1.

This is not a tolerance check, it is a physical law: the accumulated opacity along a ray is
1 - T_final, which cannot exceed 1. The shipped operator violated it on 2,635,067 of 6.5M rows
(max 2.1092), caused by catastrophic cancellation in a float32 global cumsum of log(1-alpha) --
the running sum reaches ~7e6, so recovering a local value of ~0.5 from a difference loses all
precision. The test asserts the invariant AND pins the mechanism by comparing against float64.
"""
import sys
sys.path.insert(0,"D:/Downloads/powerfoam")
import torch
from gsplat_baseline.export_gsplat_operator import _segment_exclusive_cumsum

def build(N=9_400_000, group=64, seed=0):
    g=torch.Generator().manual_seed(seed)
    alpha=(torch.rand(N,generator=g)*0.9+0.01)
    vals=torch.log((1-alpha).clamp_min(1e-12)).float()
    gs=torch.arange(0,N,group)
    gid=torch.repeat_interleave(torch.arange(gs.numel()),group)[:N]
    return alpha,vals,gs,gid

alpha,vals,gs,gid=build()
t_before=torch.exp(_segment_exclusive_cumsum(vals,gid,gs))
w=alpha*t_before
rs=torch.zeros(gs.numel()).index_add_(0,gid,w)
viol=int((rs>1.0+1e-3).sum())
print(f"rows: {gs.numel():,}   row-sum median {float(rs.median()):.6f}  max {float(rs.max()):.6f}")
print(f"rows violating sum<=1 : {viol:,}")
# exclusive cumsum must reproduce a per-group reference exactly enough to preserve the invariant
ref=torch.cat([torch.cat([torch.zeros(1,dtype=torch.float64),
      torch.cumsum(vals[s:s+64].double(),0)[:-1]]) for s in gs[:2000]])
got=_segment_exclusive_cumsum(vals,gid,gs)[:ref.numel()].double()
print(f"vs per-group reference (first 2000 groups): max abs err {float((got-ref).abs().max()):.3e}")
ok = viol==0 and float((got-ref).abs().max()) < 1e-5
print("\n"+("PASS - compositing invariant holds" if ok else "FAIL"))
sys.exit(0 if ok else 1)
