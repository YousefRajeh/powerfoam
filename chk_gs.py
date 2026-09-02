import sys; sys.path.insert(0,r"D:\Downloads\powerfoam"); sys.path.insert(0,r"D:\Downloads\powerfoam\gsplat_baseline")
import gsplat_env_gsview
import torch, configargparse, numpy as np
from configs import Params, add_group
from data_loader import DataHandler
from camera_bridge import K_from_ray_dirs
from gsplat_baseline.export_gsplat_operator import export_view_operator
cfg="output/scannet_scene0000_00_nonfrozen/config.yaml"
p=configargparse.ArgParser(); add_group(p,Params); p.add_argument("-c","--config",is_config_file=True)
args=p.parse_args(["-c",cfg]); dh=DataHandler(args); dh.reload("train",downsample=args.downsample[-1])
ck=torch.load("recon_remote/gs_froz/scene0000_00/ckpt.pt",map_location="cuda",weights_only=False)
sp=ck["splats"] if "splats" in ck else ck
means,quats=sp["means"].cuda(),sp["quats"].cuda()
scales=torch.exp(sp["scales"].cuda()); opac=torch.sigmoid(sp["opacities"].cuda().reshape(-1))
print(f"opacity: min {float(opac.min()):.3f} median {float(opac.median()):.3f} max {float(opac.max()):.3f}")
print(f"scale  : median {float(scales.median()):.4f}")
cam=dh.cameras[0]; K,_=K_from_ray_dirs(cam); K=K.cuda()
c2w=torch.eye(4,dtype=torch.float64); c2w[:3,:4]=dh.c2ws[0].double()
vm=torch.linalg.inv(c2w).float().cuda()
W,H=int(cam.width),int(cam.height)
ri,ci,v,_,acc=export_view_operator(means,quats,scales,opac,torch.zeros((means.shape[0],1),device="cuda"),
                                   vm,K,W,H,max_hits_per_pixel=512,transmittance_floor=1e-3)
rs=torch.zeros(H*W,device="cuda").index_add_(0,ri,v)
nz=torch.bincount(ri,minlength=H*W)
live=nz>0
print(f"view 0: rays with hits {int(live.sum()):,}/{H*W:,}  nnz={v.numel():,}")
print(f"  hits/ray  : mean {float(nz[live].float().mean()):.2f} max {int(nz.max())}")
print(f"  ROW SUM   : median {float(rs[live].median()):.4f}  mean {float(rs[live].mean()):.4f}  >0.99 frac {float((rs[live]>0.99).float().mean()):.3f}")
print("  -> row sum ~1 means the few hits explain ALL the light (opaque); <<1 means hits are missing")
