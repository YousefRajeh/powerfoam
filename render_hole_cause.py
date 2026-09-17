"""Do the holes in a class mask coincide with primitives the SOLVE never saw?

Renders, through PowerFoam's own rasteriser and for the same camera:
  1. the scene as it renders
  2. the class mask (per-primitive argmax, the reported rule)
  3. a CAUSE map, painting every primitive by why it could or could not be lifted:
        red     D_jj = 0        never touched by a ray -- nothing was ever lifted onto it
        orange  culled          below the opacity threshold used by the protocol
        blue    n_eff < 2       effectively a single contributing view
        green   healthy
If the holes in panel 2 land on red/blue in panel 3, the holes are a SUPPORT failure
(geometry + visibility), not a solver or a CLIP failure.
"""
from __future__ import annotations
import argparse, glob, os, sys
sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")
import numpy as np, torch, torch.nn.functional as F
import configargparse, warp as wp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from evaluate_point_cloud_miou import OPENGAUSSIAN_CLASS_SETS, embed_class_names
from diagnose_scannet_miou import load_scannet_pointcept_gt

GT_ROOT = r"D:\Downloads\scannet_pointcept"
CAUSE = {"dead (D=0)": (0.90, 0.10, 0.10), "culled": (0.95, 0.60, 0.10),
         "single view": (0.20, 0.45, 0.95), "healthy": (0.30, 0.75, 0.35)}


def lum(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0097_00")
    ap.add_argument("--recon", default="truefrozen")
    ap.add_argument("--class-set", default="opengaussian19")
    ap.add_argument("--cls", required=True)
    ap.add_argument("--views", default="0")
    ap.add_argument("--opacity-threshold", type=float, default=0.1)
    ap.add_argument("--neff-min", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"; wp.init()

    ck = f"output/scannet_{a.scene}_{a.recon}"
    p = configargparse.ArgParser(); add_group(p, Params)
    p.add_argument("-c", "--config", is_config_file=True)
    args = p.parse_args(["-c", f"{ck}/config.yaml"])
    dh = DataHandler(args); dh.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args); model.initialize_from_dataset(dh, device=dev)
    model.load_pt(f"{ck}/model.pt"); model.update_vis_cache()

    ap_ = f"artifacts/scannet/{a.scene}"
    st = torch.load(f"{ap_}/stats_{a.recon}_ogl3.pt", map_location="cpu", weights_only=False)
    D = st["support"].numpy().astype(np.float64)
    svw = st["sum_view_weight_sq"].numpy().astype(np.float64)
    n_eff = np.where(svw > 0, D ** 2 / np.maximum(svw, 1e-12), 0.0)
    sol = torch.load(f"{ap_}/solved_geometric_median_{a.recon}_ogl3.pt", map_location=dev,
                     weights_only=True)
    feats = sol["primitive_features"].to(dev).float(); valid = sol["valid_mask"].cpu().numpy()

    radii = model.get_radii().detach().cpu().numpy().reshape(-1)
    dens = model.get_density().detach().float().cpu().numpy().reshape(-1)
    alpha = 1.0 - np.exp(-dens * radii * 2.0)
    culled = alpha < a.opacity_threshold

    cand = [q for q in glob.glob(os.path.join(GT_ROOT, "*", a.scene)) if os.path.isdir(q)]
    _, raw, names = load_scannet_pointcept_gt(cand[0], "segment20")
    n2i = {n: i for i, n in enumerate(names)}; pres = set(np.unique(raw).tolist())
    kept = [n for n in OPENGAUSSIAN_CLASS_SETS[a.class_set] if n2i[n] in pres]
    assert a.cls in kept, f"'{a.cls}' not in {a.scene}: {kept}"
    k = kept.index(a.cls)
    text = embed_class_names(kept, dev)
    pred = (F.normalize(feats, dim=-1) @ text.T).argmax(1).cpu().numpy()

    cause = np.full(len(D), 3, np.int64)          # healthy
    cause[n_eff < a.neff_min] = 2
    cause[culled] = 1
    cause[D <= 0] = 0
    print("cause histogram over primitives: " +
          "  ".join(f"{n}={(cause == i).mean():.1%}" for i, n in enumerate(CAUSE)))

    c = model._vis_cache
    pts, rad = c["points"], c["radii"]
    T = model.args.num_texel_sites
    cols = torch.tensor(list(CAUSE.values()), device=dev, dtype=torch.float32)
    rows = []
    for vi in [int(v) for v in a.views.split(",")]:
        cam = dh.cameras[vi]
        with torch.no_grad():
            rgb = model.sv.forward(c["texel_sites"].view(-1, 3).detach(), cam,
                                   c["att_sites"], c["att_values"], c["att_temps"]
                                   ).view(pts.shape[0], T, 3)
            panels = [("render", rgb.clone())]
            g = 0.85 * lum(rgb).unsqueeze(-1).expand_as(rgb)
            m = g.clone()
            sel = torch.from_numpy((pred == k) & valid).to(dev)
            m[sel] = torch.tensor(CAUSE["dead (D=0)"], device=dev).view(1, 1, 3).expand(
                int(sel.sum()), T, 3) * 0 + torch.tensor([1.0, .15, .15], device=dev)
            panels.append((f"mask '{a.cls}'  ({int(sel.sum()):,} cells)", m))
            cm = cols[torch.from_numpy(cause).to(dev)].unsqueeze(1).expand(-1, T, -1)
            panels.append(("cause", cm.contiguous()))
            imgs = []
            for t, tr in panels:
                col, *_ = model.rasterizer.visualize(
                    cam, pts, rad, c["density"], c["normals"], c["texel_sites"],
                    tr.contiguous(), c["texel_height"], c["adjacency"], c["adjacency_offsets"])
                imgs.append((t, col.clamp(0, 1).cpu().numpy()))
        rows.append((vi, imgs))

    n = len(rows[0][1])
    fig, ax = plt.subplots(len(rows), n, figsize=(5.4 * n, 4.5 * len(rows)), squeeze=False)
    for r, (vi, imgs) in enumerate(rows):
        for cc, (t, im) in enumerate(imgs):
            ax[r][cc].imshow(im); ax[r][cc].set_title(f"view {vi} | {t}", fontsize=10)
            ax[r][cc].axis("off")
    ax[0][-1].legend(handles=[Patch(color=v, label=kk) for kk, v in CAUSE.items()],
                     loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{a.scene}/{a.recon} -- do mask holes coincide with unsupported primitives?",
                 fontsize=13)
    fig.tight_layout()
    out = a.out or f"artifacts/scannet/{a.scene}_holecause_{a.cls}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
