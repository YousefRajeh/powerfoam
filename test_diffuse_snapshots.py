"""The `iters` axis of the grid is free ONLY if a snapshot equals a fresh run. Verify bitwise.

If this is off by even a rounding step, every `iters` row in the grid is subtly wrong while looking
completely plausible -- so it is checked against the real `diffuse`, not reasoned about.
"""
import sys
sys.path.insert(0,"D:/Downloads/powerfoam")
import torch
torch.use_deterministic_algorithms(True)
from run_simplex_diffusion_eval import diffuse
from run_grid_search import diffuse_snapshots

dev="cuda"; P,K,E=20000,40,400000
g=torch.Generator(device=dev).manual_seed(0)
src=torch.randint(0,P,(E,),generator=g,device=dev)
dst=torch.randint(0,P,(E,),generator=g,device=dev)
keep=src!=dst; src,dst=src[keep],dst[keep]
deg=torch.zeros(P,dtype=torch.long,device=dev).index_add_(0,src,torch.ones_like(src))
p0=torch.rand((P,K),generator=g,device=dev); p0/=p0.sum(1,keepdim=True)
ok=True
for alpha in (0.5,0.95,0.99):
    snaps=diffuse_snapshots(p0,src,dst,deg,alpha,{10,30,100})
    for it in (10,30,100):
        ref=diffuse(p0,src,dst,deg,alpha,it)
        n=int((ref!=snaps[it]).sum())
        # argmax is what actually feeds the score, so report that too
        fa=int((ref.argmax(-1)!=snaps[it].argmax(-1)).sum())
        print(f"alpha={alpha:<5} iters={it:<4} elements differing={n:<10,} argmax differing={fa}")
        ok &= (n==0)
print("\n"+("PASS - snapshots are bitwise identical, iters is free" if ok else "FAIL"))
sys.exit(0 if ok else 1)
