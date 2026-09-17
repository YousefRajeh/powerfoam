"""Convert our gs_froz reconstructions into the vanilla-3DGS checkpoint LangSplat expects.

WHY THIS IS NEEDED. LangSplat's train.py refuses to start the language stage without
--start_checkpoint, and scene/gaussian_model.py::restore keys off the tuple LENGTH:
    len(model_args) == 13 -> a LangSplat feature checkpoint
    len(model_args) == 12 -> a vanilla 3DGS checkpoint   <- what we must supply
train.py additionally does `if len(model_params) == 12 and opt.include_feature: first_iter = 0`,
so a 12-tuple is exactly the supported entry point.

WHY NOT TRAIN RGB WITH LANGSPLAT ITSELF. Their train.py has an include_feature=False branch, but it
is broken in their own rasterizer: gaussian_renderer/__init__.py passes
`language_feature_precomp = torch.zeros((1,))` when the flag is off, while
cuda_rasterizer/backward.cu:494-495 loads `language_feature[coll_id * F + i]` OUTSIDE any
include_feature guard (the guard at :468 only covers dL_dpixel_F). Indexing a 1-element tensor
per-Gaussian is an out-of-bounds read, which is the "CUDA error: an illegal memory access" seen in
loss.backward() -- appearing only after densification grows coll_id enough to leave the mapped page,
hence at 800 or 1900 iterations depending on the run. Their README never trains RGB with this fork;
it starts from stock 3DGS. So we supply the checkpoint rather than patch their CUDA.

WHY gs_froz. It is the frozen arm (one primitive per GT vertex, no densification), the same budget
OpenGaussian uses, so LangSplat is compared on identical geometry rather than a reconstruction of
its own. The conventions line up exactly with LangSplat's GaussianModel -- scaling_activation=exp
against gsplat's log-scales, rotation_activation=normalize against quats, opacity_activation=sigmoid
against raw logits -- so this is a re-packing, NOT a numerical conversion.

THE OPTIMIZER SLOT IS IGNORED. restore() only calls load_state_dict when include_feature is False:
    if not training_args.include_feature:  # 以原始gs为初始化来训练feature的话，就不需要restore optimizer
        self.optimizer.load_state_dict(opt_dict)
We always run the language stage with include_feature=True, so the dict is never read. An empty one
is written to keep the tuple shape honest rather than smuggling in a foreign optimizer state.
"""
import argparse
import os

import torch

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def convert(arm, scene, out_root, iteration=30000):
    src = f"recon_remote/{arm}/{scene}/ckpt.pt"
    if not os.path.exists(src):
        return None, f"missing {src}"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck

    xyz = sp["means"].float()
    P = xyz.shape[0]
    f_dc = sp["sh0"].float()                       # (P, 1, 3)
    f_rest = sp["shN"].float()                     # (P, 15, 3)
    scaling = sp["scales"].float()                 # log-scale, matches exp activation
    rotation = sp["quats"].float()                 # normalised by the activation
    opacity = sp["opacities"].float()
    if opacity.dim() == 1:                         # LangSplat stores (P, 1)
        opacity = opacity.unsqueeze(-1)

    # sh_degree from the rest-coefficient count: (deg+1)^2 - 1 rows.
    sh_degree = int(round(((f_rest.shape[1] + 1) ** 0.5) - 1))

    # MUST be CUDA tensors. LangSplat's train.py does `torch.load(checkpoint)` with NO
    # map_location, so whatever device these were saved from is the device restore() installs them
    # on. Their own capture() always runs on cuda, so their code never needed a map_location -- but
    # we build this tuple offline, and CPU tensors here reach the rasterizer as HOST pointers:
    # rasterize_points.cu derives `radii` from means3D.options(), so preprocessCUDA writes through a
    # host pointer and dies with "illegal memory access" at rasterizer_impl.cu:276, far from the
    # actual cause. Pin the device here rather than patching their loader.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = (
        sh_degree,
        xyz.to(dev),
        f_dc.to(dev),
        f_rest.to(dev),
        scaling.to(dev),
        rotation.to(dev),
        opacity.to(dev),
        torch.zeros(P, device=dev),                # max_radii2D
        torch.zeros((P, 1), device=dev),           # xyz_gradient_accum
        torch.zeros((P, 1), device=dev),           # denom
        {},                                        # optimizer state -- never read, see docstring
        1.0,                                       # spatial_lr_scale
    )
    assert len(model_params) == 12, f"tuple must be 12 for restore(), got {len(model_params)}"

    out_dir = os.path.join(out_root, f"scannet-{scene}")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"chkpnt{iteration}.pth")
    torch.save((model_params, iteration), dst)
    return dst, f"P={P} sh_degree={sh_degree}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gs_froz", choices=("gs_froz", "gs_unfroz"))
    ap.add_argument("--out-root", default=r"D:\Downloads\baselines\LangSplat\output")
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    for s in [x for x in a.scenes.split(",") if x]:
        dst, info = convert(a.arm, s, a.out_root)
        print(f"  {s:14s} {'OK  ' + info if dst else 'FAIL ' + info}"
              f"{'  -> ' + dst if dst else ''}", flush=True)


if __name__ == "__main__":
    main()
