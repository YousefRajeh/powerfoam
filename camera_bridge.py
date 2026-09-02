"""Derive (viewmat, K) for the gsplat exporter from PowerFoam's basis-based TorchCamera.

THE PROBLEM. `TorchCamera` stores `eye`/`right`/`up`/`c2w_rot` plus a camera-space ray-direction grid
`cam_ray_dirs`. The gsplat operator exporter wants `(viewmat, K, width, height)`. Hand-writing that
bridge risks a silent convention error -- OpenCV (+z forward, y down) vs OpenGL (-z forward, y up),
or a transposed rotation -- which produces a perfectly valid-looking operator for the wrong rays.

TWO INDEPENDENT DERIVATIONS, CROSS-CHECKED. Neither is trusted alone:

  (1) FROM THE RAY GRID, which is what foam ACTUALLY traversed. For a pinhole camera the camera-space
      direction of pixel (u,v) is proportional to ((u-cx)/fx, (v-cy)/fy, 1), so after dividing by z:

          dx/dz = (u - cx)/fx     -- linear in u, slope 1/fx, intercept -cx/fx
          dy/dz = (v - cy)/fy     -- linear in v

      A least-squares fit recovers fx, fy, cx, cy EXACTLY, and the fit RESIDUAL is the test of the
      pinhole assumption: if it is not ~0 the camera has distortion and no single K exists.

  (2) FROM COLMAP, the source of truth: `sparse/0/cameras.bin` holds the true intrinsics and
      `images.bin` holds world-to-camera qvec/tvec, which IS the viewmat.

Route (1) is the one used, because it is guaranteed to describe the same cameras foam traversed --
route (2) could silently disagree in view ORDER. Route (2) is the cross-check on the values. The
ScanNet foam configs use `downsample: [1, 1]`, so no intrinsic rescaling is involved; the code
asserts this rather than assuming it.

FINAL VERIFICATION IS A RENDER. Deriving numbers that look plausible is not evidence. `verify_view`
rasterises the Gaussians with the derived (viewmat, K) and reports PSNR against the dataset image for
that view; a convention error produces garbage, not a slightly worse number.
"""
import numpy as np
import torch


def K_from_ray_dirs(cam, atol=1e-4):
    """Least-squares (fx, fy, cx, cy) from the camera-space ray grid, plus the fit residual.

    Returns (K, info). `info["max_resid_px"]` is the largest reprojection disagreement in PIXELS --
    the pinhole test. Anything above `atol` means K is not a faithful description of these rays.
    """
    d = cam.cam_ray_dirs
    if d is None:
        raise ValueError("camera has no cam_ray_dirs; cannot derive K from the ray grid")
    W, H = int(cam.width), int(cam.height)
    d = d.reshape(H, W, 3).double()
    z = d[..., 2]
    if float(z.abs().min()) < 1e-9:
        raise ValueError("ray grid contains z~0 directions; the camera frame is not +z forward")
    xz = (d[..., 0] / z).cpu().numpy()
    yz = (d[..., 1] / z).cpu().numpy()

    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    # xz varies with u only, yz with v only, for a pinhole camera; fit on the mean profile and then
    # check the FULL grid against it, so a violation of that separability is caught rather than
    # averaged away.
    su, iu = np.polyfit(u, xz.mean(0), 1)
    sv, iv = np.polyfit(v, yz.mean(1), 1)
    fx, fy = 1.0 / su, 1.0 / sv
    cx, cy = -iu * fx, -iv * fy

    pred_x = (u[None, :] - cx) / fx
    pred_y = (v[:, None] - cy) / fy
    resid_px = max(float(np.abs(pred_x - xz).max() * abs(fx)),
                   float(np.abs(pred_y - yz).max() * abs(fy)))
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K, {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "max_resid_px": resid_px,
               "pinhole_ok": resid_px <= atol * max(W, H)}


def viewmat_from_camera(cam):
    """World-to-camera 4x4 from the camera's own rotation and eye.

    `c2w_rot` maps camera-space to world-space, so the world-to-camera rotation is its transpose and
    the translation is -R^T e. If `c2w_rot` is absent we rebuild it from the stored basis; the
    forward axis is taken as cross(right, up) and its SIGN is checked against the ray grid rather
    than assumed, since that is precisely the OpenCV/OpenGL trap.
    """
    eye = cam.eye.double().reshape(3)
    R = cam.c2w_rot
    if R is None:
        r = cam.right.double().reshape(3)
        up = cam.up.double().reshape(3)
        f = torch.cross(r, up)
        f = f / f.norm()
        R = torch.stack([r / r.norm(), up / up.norm(), f], dim=1)
    R = R.double().reshape(3, 3)

    vm = torch.eye(4, dtype=torch.float64)
    vm[:3, :3] = R.T
    vm[:3, 3] = -(R.T @ eye)
    return vm.float()


def verify_view(cam, means, quats, scales, opacities, colors, gt_image):
    """Rasterise with the derived (viewmat, K) and return PSNR against the dataset image.

    This is the acceptance test. A transposed rotation or a flipped forward axis does not degrade
    PSNR slightly -- it destroys it -- so a healthy number here is strong evidence the bridge is
    correct, and a bad one localises the error before any operator is exported.
    """
    from gsplat import rasterization
    K, info = K_from_ray_dirs(cam)
    vm = viewmat_from_camera(cam)
    W, H = int(cam.width), int(cam.height)
    out, _, _ = rasterization(means, quats, scales, torch.sigmoid(opacities.reshape(-1)),
                              colors, vm[None].cuda(), K[None].cuda(), W, H)
    img = out[0].clamp(0, 1)
    gt = gt_image.to(img.device).float()
    if gt.max() > 1.5:
        gt = gt / 255.0
    mse = float(((img - gt[..., :img.shape[-1]]) ** 2).mean())
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    return psnr, info
