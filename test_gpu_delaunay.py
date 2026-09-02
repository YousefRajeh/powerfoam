"""GPU Delaunay must produce the SAME EDGE SET as scipy, not merely a plausible-looking graph.

The dangerous failure here is the permutation: radfoam re-sorts points internally, so forgetting
to map slots back through permutation() gives a graph with the right degree distribution, no
self-loops and full symmetry -- it passes every structural check while being a triangulation of
scrambled identities. Only comparing the actual edge SET against an independent implementation
catches that, so this test compares to scipy on sizes small enough for scipy to finish.
"""
import sys
sys.path.insert(0,"D:/Downloads/powerfoam")
import numpy as np, torch
from scipy.spatial import Delaunay
from graph_variants import delaunay_graph

def edgeset(src,dst):
    a=torch.minimum(src,dst).cpu().numpy(); b=torch.maximum(src,dst).cpu().numpy()
    return set(map(tuple,np.unique(np.stack([a,b],1),axis=0)))

def scipy_edges(pts, idx, P):
    tri=Delaunay(pts.astype(np.float64)); simp=tri.simplices
    pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    e=np.concatenate([simp[:,list(p)] for p in pairs],0)
    e=idx[e]
    a=np.minimum(e[:,0],e[:,1]); b=np.maximum(e[:,0],e[:,1])
    return set(map(tuple,np.unique(np.stack([a,b],1),axis=0)))

ok=True
for N,seed in ((2000,0),(20000,1),(60000,2)):
    g=torch.Generator(device="cuda").manual_seed(seed)
    P=N+37                                     # extra invalid rows so index mapping is exercised
    pos=torch.rand((P,3),generator=g,device="cuda")*10.0
    vm=torch.zeros(P,dtype=torch.bool,device="cuda")
    sel=torch.randperm(P,generator=g,device="cuda")[:N]; vm[sel]=True
    src,dst,_=delaunay_graph(pos,vm,device="cuda",backend="gpu")
    idx=torch.nonzero(vm).squeeze(1)
    E_gpu=edgeset(src,dst)
    E_cpu=scipy_edges(pos[idx].cpu().numpy(), idx.cpu().numpy(), P)
    inter=len(E_gpu&E_cpu); only_g=len(E_gpu-E_cpu); only_c=len(E_cpu-E_gpu)
    jac=inter/max(len(E_gpu|E_cpu),1)
    # structural checks that a permutation bug would still pass -- reported to show they are
    # insufficient on their own
    selfloop=int((src==dst).sum()); valid=bool(vm[src].all() and vm[dst].all())
    print(f"N={N:6d}  gpu_edges={len(E_gpu):8,} scipy={len(E_cpu):8,}  "
          f"shared={inter:8,}  gpu_only={only_g:6,}  scipy_only={only_c:6,}  Jaccard={jac:.6f}")
    print(f"          [structural: self-loops={selfloop} all-endpoints-valid={valid}]")
    ok &= (jac > 0.999)
print("\n"+("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
