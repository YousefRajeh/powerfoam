"""Snapshot accumulator outputs under the CURRENT implementation, so the in-place rewrite that
follows can be compared against it bitwise. Without this, a refactor of accumulate_view has no
reference at all -- test_lean_stats compares lean vs full WITHIN one version, which cannot detect a
change that moves both."""
import sys, torch
sys.path.insert(0,"D:/Downloads/feature-foam-lifting/src")
torch.use_deterministic_algorithms(True)
from feature_foam_lifting.operator import AccumulatedFeatureStats
DEV="cuda"; P,F,V=5000,512,12
def build(lean,seed=0):
    st=AccumulatedFeatureStats.zeros(P,F,device=DEV,lean=lean)
    g=torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(V):
        nnz=int(torch.randint(2000,6000,(1,),generator=g,device=DEV).item())
        cols=torch.randint(0,P,(nnz,),generator=g,device=DEV)
        vals=torch.rand(nnz,generator=g,device=DEV)+1e-3
        b=torch.nn.functional.normalize(torch.randn(nnz,F,generator=g,device=DEV),dim=-1)
        st.accumulate_view(cols,vals,b)
    return st
out={}
for lean in (False,True):
    st=build(lean)
    tag="lean" if lean else "full"
    for k in ("numerator","support","gm_z","gm_weight","intra_sum","sum_view_weight_sq"):
        out[f"{tag}.{k}"]=getattr(st,k).cpu()
    r=st.reliability()
    for k,v in r.items():
        if torch.is_tensor(v): out[f"{tag}.rel.{k}"]=v.cpu()
torch.save(out,"artifacts/accum_reference_preinplace.pt")
print("saved",len(out),"tensors")
