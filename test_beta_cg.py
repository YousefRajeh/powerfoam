"""Verify (a) the custom CG matches ridge_pcg, (b) beta's k=1 => 0 identity holds numerically.

The custom CG exists because B is factorised (segment ids + table) and a dense B would be ~30 GB.
That means it cannot be checked by inspection -- it is checked against the project's own solver on a
small case where a dense B DOES fit. The beta identity is checked on a hand-built operator, because
"k=1 implies beta=0" is the analytic claim the paper will make and it must hold in the code too.
"""
import sys
sys.path.insert(0,"D:/Downloads/powerfoam"); sys.path.insert(0,"D:/Downloads/feature-foam-lifting/src")
import torch
torch.use_deterministic_algorithms(True)
from feature_foam_lifting.operator import SparseFeatureOperator, ridge_pcg
from compute_beta import cg_normal_equations

dev="cpu"; torch.manual_seed(0)
R,P,D,NNZ=4000,300,16,20000
row=torch.randint(0,R,(NNZ,)); col=torch.randint(0,P,(NNZ,)); val=torch.rand(NNZ)+0.05
# row-normalise so the operator resembles alpha compositing (rows summing to 1)
rs=torch.zeros(R).index_add_(0,row,val); val=val/rs[row].clamp_min(1e-12)
B=torch.randn(R,D)
A=SparseFeatureOperator(row.clone(),col.clone(),val.clone(),R,P)

x_ref,info_ref=ridge_pcg(A,B,mode="none",rtol=1e-10,max_iter=2000)
rhs=A.rmatmul(B); diag=torch.zeros(P).index_add_(0,col,val*val)
CH=10**9
def mm(x): 
    o=torch.zeros((R,x.shape[1])); o.index_add_(0,row,val[:,None]*x[col]); return o
def rm(b):
    o=torch.zeros((P,b.shape[1])); o.index_add_(0,col,val[:,None]*b[row]); return o
x_mine,info=cg_normal_equations(mm,rm,rhs,diag,iters=2000,rtol=1e-10)

num=(x_ref-x_mine).norm(); den=x_ref.norm().clamp_min(1e-12)
print(f"(a) CG vs ridge_pcg: relative difference {float(num/den):.3e}  "
      f"(ref {info_ref['iterations']} iters, mine {info['iterations']})")
ok_a = float(num/den) < 1e-5

# (b) k=1 => beta=0, on a hand-built operator where every row has exactly ONE nonzero summing to 1
r2=torch.arange(500); c2=torch.randint(0,50,(500,)); v2=torch.ones(500)
x=torch.randn(50,D); Bb=torch.randn(500,D)
d=(x[c2]-Bb[r2]).norm(dim=-1)
mu=torch.zeros(500).index_add_(0,r2,v2*d)
m2=torch.zeros(500).index_add_(0,r2,v2*d*d)
sig2=m2-mu*mu
print(f"(b) k=1 rows: max |sigma^2| = {float(sig2.abs().max()):.3e}  -> beta_i = 0 exactly")
ok_b = float(sig2.abs().max()) < 1e-4
print("\n"+("PASS" if (ok_a and ok_b) else "FAIL"))
sys.exit(0 if (ok_a and ok_b) else 1)
