"""Prepare OccamLGS / VALA inputs for a ScanNet scene from a gsplat reconstruction.

WHAT THESE METHODS NEED (read from gaussian_feature_extractor.py and scene/__init__.py):
  1. <model>/point_cloud/iteration_30000/point_cloud.ply -- Scene(load_iteration=30000) loads this
  2. <model>/chkpnt30000.pth                              -- restore_rgb() reads the 12-element tuple
  3. <model>/cfg_args                                     -- get_combined_args() parses it
  4. language features, 512-d -- NO autoencoder for either method (unlike LangSplat).
     Occam reads <source>/language_features, VALA reads <source>/langsplat/language_features.
  5. <source>/images, <source>/sparse                     -- standard COLMAP layout

WHY BUILD THE CHECKPOINT THROUGH THEIR OWN GaussianModel RATHER THAN HAND-ROLLING THE TUPLE:
restore_rgb() ends with `self.optimizer.load_state_dict(opt_dict)` -- UNCONDITIONALLY, unlike
LangSplat's restore() which skips it when training features. make_langsplat_ckpt.py writes
`opt_dict = {}` (LangSplat never reads it), and an empty dict makes Adam.load_state_dict raise. So
here we instantiate their GaussianModel, install our tensors, call their training_setup() to build a
real Adam, and serialise with their capture_rgb(). The optimizer state is then genuine and the tuple
is exactly what restore_rgb expects -- no edit to their code.

Tensors must land on CUDA: capture_rgb stores them as-is and restore_rgb installs them directly, so
CPU tensors would reach the rasterizer as host pointers (the failure mode already hit in LangSplat
stage 3).

RESOLUTION: both index seg_map[:, self.y, self.x] with y/x built from the loaded image size -- they
do NOT resample. The images must stay at the resolution the seg maps were extracted at (968x1296
here), so resolution=-1 and no -r flag. A downscaled image silently reads the top-left corner of the
seg map, which is what invalidated the first LangSplat stage-3 sweep.

VALA is a fork of OccamLGS (same GaussianModel, same 12-tuple, same gsplat fork), so one script
serves both; BASELINE_REPO picks whose code builds the checkpoint and BASELINE_CKPT_GLOB locates the
reconstruction (995 keeps it under ~/gaussian_baseline_scannet/<scene>/ckpts/).
"""
import argparse
import glob
import os
import sys
import torch
from torch import nn

REPO = os.environ.get("BASELINE_REPO", r"D:\Downloads\baselines\OccamLGS")
sys.path.insert(0, REPO)

DEFAULT_GLOB = os.path.join(r"D:\Downloads\powerfoam", "recon_remote", "{arm}", "{scene}", "ckpt.pt")
CKPT_GLOB = os.environ.get("BASELINE_CKPT_GLOB", DEFAULT_GLOB)

SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
          "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00"]


def build(arm, scene, model_root, source_root, feature_level, iteration=30000):
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams
    from argparse import ArgumentParser as AP

    pattern = CKPT_GLOB.format(arm=arm, scene=scene)
    cands = sorted(glob.glob(pattern))
    if not cands:
        return None, f"no ckpt matching {pattern}"
    src = cands[-1]

    ck = torch.load(src, map_location="cpu", weights_only=False)
    sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck

    xyz = sp["means"].float().cuda()
    P = xyz.shape[0]
    f_dc = sp["sh0"].float().cuda()
    f_rest = sp["shN"].float().cuda()
    scaling = sp["scales"].float().cuda()
    rotation = sp["quats"].float().cuda()
    opacity = sp["opacities"].float().cuda()
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)
    sh_degree = int(round(((f_rest.shape[1] + 1) ** 0.5) - 1))

    g = GaussianModel(sh_degree)
    g.active_sh_degree = sh_degree
    g._xyz = nn.Parameter(xyz.requires_grad_(True))
    g._features_dc = nn.Parameter(f_dc.requires_grad_(True))
    g._features_rest = nn.Parameter(f_rest.requires_grad_(True))
    g._scaling = nn.Parameter(scaling.requires_grad_(True))
    g._rotation = nn.Parameter(rotation.requires_grad_(True))
    g._opacity = nn.Parameter(opacity.requires_grad_(True))
    g.max_radii2D = torch.zeros(P, device="cuda")
    g.spatial_lr_scale = 1.0

    p = AP()
    op = OptimizationParams(p)
    opt = op.extract(p.parse_args([]))
    g.training_setup(opt)                      # builds the real Adam that capture_rgb serialises

    model_path = os.path.join(model_root, f"scannet-{scene}")
    pc_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(pc_dir, exist_ok=True)
    g.save_ply(os.path.join(pc_dir, "point_cloud.ply"))
    torch.save((g.capture_rgb(), iteration), os.path.join(model_path, f"chkpnt{iteration}.pth"))

    # Every field ModelParams defines must appear -- get_combined_args merges this Namespace under
    # the CLI, and any omission surfaces later as an AttributeError deep inside the COLMAP loader
    # (e.g. 'GroupParams' object has no attribute 'depths').
    #
    # eval=False so every camera is a training camera. VALA's own run_scannet.sh passes --eval,
    # which holds out every 8th view; we keep all views across every baseline instead, because the
    # view set is shared experimental setup rather than a method choice, and mixing 7/8 with 8/8
    # would confound the comparison. Documented deviation.
    source_path = os.path.join(source_root, scene)
    fields = dict(
        sh_degree=sh_degree, source_path=source_path.replace("\\", "/"),
        model_path=model_path.replace("\\", "/"), images="images", depths="",
        resolution=-1, white_background=False, train_test_exp=False,
        data_device="cuda", eval=False, language_features_name="language_features",
        feature_level=feature_level,
    )
    with open(os.path.join(model_path, "cfg_args"), "w") as fh:
        fh.write("Namespace(" + ", ".join(f"{k}={v!r}" for k, v in sorted(fields.items())) + ")")
    return model_path, f"P={P} sh_degree={sh_degree} level={feature_level}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gs_froz", choices=("gs_froz", "gs_unfroz"))
    ap.add_argument("--model-root", default=os.path.join(REPO, "output"))
    ap.add_argument("--source-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--scenes", default=",".join(SCENES))
    # Occam's code default is 2; VALA's run_scannet.sh specifies 0 for ScanNet.
    ap.add_argument("--feature-level", type=int, default=2)
    a = ap.parse_args()
    for s in [x for x in a.scenes.split(",") if x]:
        mp, info = build(a.arm, s, a.model_root, a.source_root, a.feature_level)
        print(f"  {s:14s} {'OK  ' + info if mp else 'FAIL ' + info}", flush=True)


if __name__ == "__main__":
    main()
