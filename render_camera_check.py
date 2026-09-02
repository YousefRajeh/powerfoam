"""Visual verification of the camera bridge: ground truth beside a render with the DERIVED camera.

PSNR can be argued with; a picture cannot. If `viewmat`/`K` derived from PowerFoam's basis-based
TorchCamera are correct, the right panel shows the same room from the same viewpoint as the left.
A transposed rotation or a flipped forward axis (the OpenCV/OpenGL trap) produces noise or an empty
frame, not a slightly worse image.

Writes artifacts/scannet/camera_bridge_check.png -- three views stacked, GT | render side by side.
"""
import sys

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\powerfoam\gsplat_baseline")

# MUST precede `import gsplat`: gsplat_cuda links against the gs-view env's torch DLLs and CUDA 11.8,
# neither of which is on this interpreter's DLL search path, so the import fails with
# "DLL load failed while importing gsplat_cuda" without it.
import gsplat_env_gsview  # noqa: F401

import configargparse
import numpy as np
import torch
from PIL import Image

from camera_bridge import K_from_ray_dirs, viewmat_from_camera
from configs import Params, add_group
from data_loader import DataHandler
from gsplat import rasterization

SCENE = "scene0000_00"
VIEWS = (0, 80, 160)
OUT = "artifacts/scannet/camera_bridge_check.png"


def gt_image(dh, i):
    """The dataset image for view i as uint8 RGB. It lives in `dh.rgbs` (float 0-1) -- the earlier
    guess at `images`/`all_images` silently returned None, so PSNR was computed against a BLACK
    frame and reported ~5 dB for a render that may have been fine."""
    a = dh.rgbs[i].detach().cpu().numpy()[..., :3]
    return (a * 255).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)


def main():
    cfg = f"output/scannet_{SCENE}_truefrozen/config.yaml"
    p = configargparse.ArgParser()
    add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", cfg])
    dh = DataHandler(args)
    dh.reload("train", downsample=args.downsample[-1])

    ck = torch.load(f"recon_remote/gs_froz/{SCENE}/ckpt.pt", map_location="cuda",
                    weights_only=False)
    sp = ck["splats"] if "splats" in ck else ck
    means, quats = sp["means"].cuda(), sp["quats"].cuda()
    scales = torch.exp(sp["scales"].cuda())
    opac = torch.sigmoid(sp["opacities"].cuda().reshape(-1))
    sh = torch.cat([sp["sh0"].cuda(), sp["shN"].cuda()], 1)

    panels = []
    for vi in VIEWS:
        cam = dh.cameras[vi]
        K, info = K_from_ray_dirs(cam)
        # world-to-camera straight from the loader's own camera-to-world matrices. `c2ws` is what
        # the dataset actually uses (viewer_forward = c2ws[:, :3, 2], i.e. column 2 is forward --
        # the OpenCV convention gsplat expects), so no basis reconstruction and no sign guessing.
        # c2ws are 3x4 (rotation | translation); pad to 4x4 before inverting.
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :4] = dh.c2ws[vi].double()
        vm = torch.linalg.inv(c2w).float()
        W, H = int(cam.width), int(cam.height)
        out, _, _ = rasterization(means, quats, scales, opac, sh,
                                  vm[None].cuda(), K[None].cuda(), W, H, sh_degree=3)
        img = out[0].clamp(0, 1)
        g = gt_image(dh, vi)
        rgb = (img * 255).byte().cpu().numpy()
        if g is None:
            g = np.zeros_like(rgb)
        mse = float(((img.cpu() - torch.from_numpy(g / 255.0).float()) ** 2).mean())
        print(f"view {vi:4d}: PSNR {10*np.log10(1/max(mse,1e-12)):6.2f} dB   "
              f"fx={info['fx']:.1f} cx={info['cx']:.1f}  pinhole_resid={info['max_resid_px']:.1e}px",
              flush=True)
        panels.append(np.concatenate([g, rgb], axis=1))

    Image.fromarray(np.concatenate(panels, axis=0)).save(OUT)
    print(f"saved {OUT}   (LEFT = ground truth, RIGHT = render with the derived camera)")


if __name__ == "__main__":
    main()
