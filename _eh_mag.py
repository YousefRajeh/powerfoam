import sys, torch, numpy as np
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src"); sys.path.insert(0, r"D:\Downloads\powerfoam")
from determinism import enable_determinism; enable_determinism(verbose=False)
import measure_xball2 as XB
dev='cuda'
print(f"{'scene':14s} {'E_H':>12s} {'graph':>12s} {'extra':>12s} {'extra/E_H':>10s} {'c':>12s} {'mean r':>9s}", flush=True)
for sc in ('scene0000_00','scene0070_00','scene0400_00'):
    row,col,val,gid,Treg,P,R,_ = XB.build(sc,'pf_truefrozen',12,512,dev)
    o=torch.argsort(row); row,col,val=row[o].contiguous(),col[o].contiguous(),val[o].contiguous()
    nnz=val.numel(); d=Treg.shape[1]
    D=torch.zeros(P,device=dev).index_add_(0,col,val)
    rvec=torch.zeros(R,device=dev).index_add_(0,row,val)
    Atr=torch.zeros(P,device=dev).index_add_(0,col,val*rvec[row])
    rhs=torch.zeros((P,d),device=dev)
    for s0 in range(0,nnz,8_000_000):
        e0=min(s0+8_000_000,nnz); rhs.index_add_(0,col[s0:e0],val[s0:e0,None]*Treg[gid[row[s0:e0]]])
    X=rhs/D.clamp_min(torch.finfo(val.dtype).eps)[:,None]
    xn2=(X**2).sum(1)
    starts=torch.searchsorted(row,torch.arange(R+1,device=dev)); _st=starts.cpu().tolist()
    BUD=max(1,int(3e8//d)); tg=torch.arange(0,nnz+BUD,BUD,device=dev)
    bnd=torch.unique(torch.cat([torch.searchsorted(starts.contiguous(),tg).clamp(0,R),torch.tensor([R],device=dev)]))
    blocks=[(int(a),int(b),_st[int(a)],_st[int(b)]) for a,b in zip(bnd[:-1],bnd[1:]) if _st[int(b)]>_st[int(a)]]
    AX2=0.0
    for r0,r1,s_,e_ in blocks:
        lr=row[s_:e_]-r0; cs=col[s_:e_]; vw=val[s_:e_,None]
        t=torch.zeros((r1-r0,d),device=dev); t.index_add_(0,lr,vw*X[cs]); AX2+=float((t*t).sum()); del t
    EH=float((D*xn2).sum())-AX2
    graph=float((Atr*xn2).sum())-AX2
    extra=float(((D-Atr)*xn2).sum())
    hit=torch.zeros(R,dtype=torch.bool,device=dev); hit[row]=True; idx=hit.nonzero(as_tuple=True)[0]
    c=float(((1.0-rvec[idx])*(Treg[gid[idx]]**2).sum(1)).sum())
    print(f"{sc:14s} {EH:12.1f} {graph:12.1f} {extra:12.1f} {100*extra/max(EH,1e-9):9.2f}% {c:12.1f} {float(rvec[idx].mean()):9.6f}", flush=True)
    del row,col,val,gid,Treg,rhs,X; torch.cuda.empty_cache()
