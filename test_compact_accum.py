"""Verify the compact (columns_unique) path is BITWISE identical to the dense path.

The compact path skips the (P,) and (P, F) per-view scatter buffers, which is valid only when each
primitive appears at most once per view -- true for distill.py's replay, where `ids` are unique
(NormLift's own `self.splat_features[ids] += ...` relies on the same property; indexed += does not
accumulate duplicates). It reproduces the dense arithmetic exactly, including computing
view_feature as (v*b)/v rather than b.

This duplicates the geometric-median update, so it could silently drift from the dense one later --
this test is the guard against that, and it checks the gm accumulators specifically.
"""
import sys, torch
sys.path.insert(0,"D:/Downloads/feature-foam-lifting/src")
torch.use_deterministic_algorithms(True)
from feature_foam_lifting.operator import AccumulatedFeatureStats
DEV="cuda"; P,F,V=5000,512,16

def build(compact, lean, seed=0):
    st=AccumulatedFeatureStats.zeros(P,F,device=DEV,lean=lean)
    g=torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(V):
        # UNIQUE columns per view -- the premise of the compact path. Sizes vary per view so the
        # init/update split of the geometric median is genuinely exercised.
        n=int(torch.randint(P//4,P,(1,),generator=g,device=DEV).item())
        cols=torch.randperm(P,generator=g,device=DEV)[:n]
        vals=torch.rand(n,generator=g,device=DEV)
        vals[torch.rand(n,generator=g,device=DEV)<0.05]=0.0   # exercise the untouched branch
        b=torch.nn.functional.normalize(torch.randn(n,F,generator=g,device=DEV),dim=-1)
        st.accumulate_view(cols,vals,b,columns_unique=compact)
    return st

ok=True
for lean in (False,True):
    dense,comp=build(False,lean),build(True,lean)
    fields=("numerator","support","gm_z","gm_weight","intra_sum","sum_view_weight_sq")
    bad=[f"{k}:{int((getattr(dense,k)!=getattr(comp,k)).sum())}"
         for k in fields if int((getattr(dense,k)!=getattr(comp,k)).sum())]
    rd,rc=dense.reliability(),comp.reliability()
    bad+= [f"rel.{k}:{int((rd[k]!=rc[k]).sum())}" for k in rd
           if torch.is_tensor(rd[k]) and int((rd[k]!=rc[k]).sum())]
    nz=int((dense.gm_weight>0).sum())
    print(f"lean={lean!s:5s} gm-active primitives={nz:,}  -> "
          f"{'ALL BITWISE IDENTICAL' if not bad else 'DIFFERS ['+', '.join(bad)+']'}")
    ok&=not bad
print("\n"+("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
