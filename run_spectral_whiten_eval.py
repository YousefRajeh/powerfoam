"""Spectral whitening of lifted features: generalise centering from 1 direction to k.

Centering removes the mean direction and gains +2.25 mIoU -- the DC-removal step, in cryo-ET terms.
But the mean is only the top eigenvector of the feature covariance. If further high-variance
directions are also non-discriminative (shared scene/domain structure rather than class content),
downweighting them should extend the gain. That is the whitening step: equalise power across
directions so variance stops standing in for information.

    f' = normalize( f - sum_{i<k} beta_i <f, v_i> v_i )      v_i = top-k PCs of the lifted features

beta = 1 removes a component entirely; beta < 1 shrinks it. k=1 with beta=lam reproduces centering.
Uses only the features themselves (no labels), and leaves the benchmark's bare cosine argmax intact.
"""
import argparse, json, os, sys, time
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src"); sys.path.insert(0, r"D:\Downloads\powerfoam")
import numpy as np, torch, torch.nn.functional as F
from determinism import enable_determinism
from evaluate_point_cloud_miou import (OPENGAUSSIAN_CLASS_SETS, embed_class_names,
    calculate_metrics, remap_gt_labels, load_scannet_pointcept_gt)
from point_cloud_query import assign_points_to_power_cells
from run_cluster_classify_eval import SCENES, CLASS_SETS
from build_true_facet_graph import load_points_radii
from run_simplex_diffusion_eval import HARDEST_FIRST

p = argparse.ArgumentParser()
p.add_argument("--scenes", default=",".join(HARDEST_FIRST))
p.add_argument("--variant", default="nonfrozen"); p.add_argument("--suffix", default="_ogl3")
p.add_argument("--gt-root", default=r"D:\Downloads\scannet_pointcept")
p.add_argument("--ks", default="1,2,3,5,10"); p.add_argument("--betas", default="0.3,0.5")
p.add_argument("--outdir", default="artifacts/scannet/whiten")
a = p.parse_args(); enable_determinism(); os.makedirs(a.outdir, exist_ok=True); dev="cuda"
ks=[int(x) for x in a.ks.split(",")]; betas=[float(x) for x in a.betas.split(",")]

for scene in a.scenes.split(","):
    op=os.path.join(a.outdir,f"{scene}.json")
    if os.path.exists(op): print(f"[skip] {scene}",flush=True); continue
    t0=time.time(); split=SCENES[scene]; art=f"artifacts/scannet/{scene}"
    centers,radii=load_points_radii(f"output/scannet_{scene}_{a.variant}")
    s=torch.load(f"{art}/solved_geometric_median_{a.variant}{a.suffix}.pt",map_location=dev,weights_only=True)
    f=s["primitive_features"].float(); vmn=s["valid_mask"].cpu().numpy(); vm=torch.from_numpy(vmn).to(dev)
    P=f.shape[0]; u=torch.zeros_like(f); u[vm]=F.normalize(f[vm],dim=-1); del f,s
    mu=F.normalize(u[vm].mean(0,keepdim=True),dim=-1)
    # PCs of the MEAN-CENTRED features: v_1 is then the top residual direction, not the mean itself
    Xc=u[vm]-u[vm].mean(0,keepdim=True)
    idx=torch.randperm(Xc.shape[0],device=dev)[:120000]
    _,S_,V=torch.linalg.svd(Xc[idx],full_matrices=False)
    ev=(S_**2); ev=ev/ev.sum()
    print(f"[{scene}] explained var of top PCs: "+" ".join(f"{float(ev[i]):.3f}" for i in range(6)),flush=True)
    gt,raw,alln=load_scannet_pointcept_gt(f"{a.gt_root}/{split}/{scene}","segment20")
    asg=assign_points_to_power_cells(gt,centers,radii,valid=vmn,k=64); own=asg>=0
    n2i={n:i for i,n in enumerate(alln)}; pres=set(np.unique(raw).tolist())
    res={"scene":scene,"explained_var":[float(x) for x in ev[:10]],"arms":{}}
    for cs in CLASS_SETS:
        kept=[(n2i[n],n) for n in OPENGAUSSIAN_CLASS_SETS[cs] if n2i[n] in pres]
        tids=[i for i,_ in kept]; names=[n for _,n in kept]
        gt_t=torch.from_numpy(remap_gt_labels(raw,tids)).long(); T=embed_class_names(names,dev)
        def sc(c,tag):
            pr=np.zeros(gt.shape[0],dtype=np.int64); pr[own]=c[asg[own]]+1
            _,m,_,ma=calculate_metrics(gt_t,torch.from_numpy(pr).long(),len(tids)+1)
            res["arms"].setdefault(tag,{})[cs]={"mIoU":float(m)*100,"mAcc":float(ma)*100}; return float(m)*100
        b=sc((u@T.T).argmax(-1).cpu().numpy(),"plain"); print(f"  {cs} [plain] {b:.2f}",flush=True)
        cf=torch.zeros_like(u); cf[vm]=F.normalize(u[vm]-0.3*mu,dim=-1)
        v=sc((cf@T.T).argmax(-1).cpu().numpy(),"center_lam0.3"); print(f"  {cs} [center_lam0.3] {v:.2f} ({v-b:+.2f})",flush=True)
        for k in ks:
            Vk=V[:k]
            for be in betas:
                cf=torch.zeros_like(u)
                x=u[vm]-0.3*mu
                cf[vm]=F.normalize(x-be*((x@Vk.T)@Vk),dim=-1)
                v=sc((cf@T.T).argmax(-1).cpu().numpy(),f"whiten_k{k}_b{be:g}")
                print(f"  {cs} [whiten_k{k}_b{be:g}] {v:.2f} ({v-b:+.2f})",flush=True)
                del cf
        del T
    json.dump(res,open(op,"w"),indent=2); print(f"[{scene}] done {time.time()-t0:.0f}s",flush=True)
