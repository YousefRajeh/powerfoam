"""Verify the in-place accumulate_view rewrite is BITWISE identical to the pre-rewrite version.

The rewrite reuses view_numerator as view_feature and then as f_v, replacing three (P, F)
allocations and their boolean-mask gathers with two in-place divisions. That is an exact algebraic
no-op -- but "exact on paper" is not evidence, and this function feeds every solved feature in the
project (foam included), so it is checked against outputs captured from the ORIGINAL code
(artifacts/accum_reference_preinplace.pt) rather than against a fresh run of itself.
"""
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
ref=torch.load("artifacts/accum_reference_preinplace.pt",weights_only=True)
ok=True
for lean in (False,True):
    st=build(lean); tag="lean" if lean else "full"
    r=st.reliability()
    fields={k:getattr(st,k).cpu() for k in
            ("numerator","support","gm_z","gm_weight","intra_sum","sum_view_weight_sq")}
    fields.update({f"rel.{k}":v.cpu() for k,v in r.items() if torch.is_tensor(v)})
    bad=[]
    for k,v in fields.items():
        rv=ref[f"{tag}.{k}"]
        n=int((v!=rv).sum())
        if n: bad.append(f"{k}:{n}")
    print(f"{tag:5s}: {len(fields)} tensors checked -> "
          f"{'ALL BITWISE IDENTICAL' if not bad else 'DIFFERS ['+', '.join(bad)+']'}")
    ok &= not bad
print("\n"+("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
