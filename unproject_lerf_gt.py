"""Build a 3D point-level ground truth for a LERF-OVS scene, the same idea as
unproject_replica_gt.py but for LERF-OVS's sparse per-frame polygon GT instead of Replica's
dense per-pixel semantic renders.

No COLMAP dense-stereo depth exists for these scenes (checked: stereo/depth_maps only has
fusion.cfg/patch-match.cfg, no actual depth maps -- dense MVS was never run). Rather than
integrating a new monocular depth estimator (its own scale/convention risk, as room_0's
back-projection just demonstrated), this reuses each METHOD'S OWN rendered depth at the labeled
camera pose -- consistent with this project's established principle of giving each method its
own native mechanism rather than reimplementing one method's approach on the other's terms.
This script handles PowerFoam's side (forward_visualization's ray-marched depth, same as the
viewer uses); the Splat Feature Solver side needs its own equivalent depth-render call.

Ray convention verified directly against the renderer (not assumed): `rasterize.py`'s
`depth_out[pix_i,pix_j] = depth_quantile_out`, built from `t_near` (a ray-sphere intersection
distance), and `TorchCamera._build_pinhole_ray_maps` already builds exactly this per-pixel
normalized-ray-direction convention (`get_ray_dir` then normalized) for the SAME renderer to
consume -- so depth is real Euclidean distance along that normalized ray from `camera.eye`, and
`world_point = eye + depth * normalized_ray_dir` reuses the renderer's own ray formula rather
than re-deriving a pinhole projection from scratch (the mistake that caused the room_0 bug).
Background/no-hit pixels get depth=0.0 (confirmed: `depth_quantile_out` initializes at 0.0 and
is only overwritten on a found intersection) -- excluded via depth > 0.

Polygon labels: LERF-OVS's label JSONs store `segmentation` as a literal list of [x, y] polygon
vertices (not COCO's nested/flat format -- checked directly). Rasterized via PIL at the camera's
actual working resolution (scaling polygon coordinates from the JSON's own recorded width/height
to the camera's), painter's-algorithm ordered by each object's `layer` field.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")


def load_scene_labels(scene):
    label_dir = LERF_ROOT / "label" / scene
    frames = []
    categories = set()
    for f in sorted(label_dir.glob("*.json")):
        info = json.loads(f.read_text())
        frames.append(info)
        for o in info["objects"]:
            categories.add(o["category"])
    category_list = sorted(categories)
    cat_to_id = {c: i + 1 for i, c in enumerate(category_list)}
    return frames, cat_to_id


def rasterize_frame_labels(info, cat_to_id, out_w, out_h):
    json_w, json_h = info["info"]["width"], info["info"]["height"]
    sx, sy = out_w / json_w, out_h / json_h
    canvas = Image.new("I", (out_w, out_h), 0)
    draw = ImageDraw.Draw(canvas)
    objects = sorted(info["objects"], key=lambda o: o.get("layer", 0))
    for o in objects:
        poly = [(x * sx, y * sy) for x, y in o["segmentation"]]
        if len(poly) < 3:
            continue
        draw.polygon(poly, fill=cat_to_id[o["category"]])
    return np.array(canvas, dtype=np.int64)


def unproject_labeled_frame(camera, depth, label_canvas, device):
    camera = camera.to_device(device)
    h, w = camera.height, camera.width
    ii, jj = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ray_dirs = camera.get_ray_dir(ii.reshape(-1), jj.reshape(-1))
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)

    depth_flat = depth.reshape(-1).to(device)
    label_flat = torch.from_numpy(label_canvas).reshape(-1).to(device)
    valid = (depth_flat > 0) & (label_flat > 0)
    if not torch.any(valid):
        return None, None

    world_pts = camera.eye[None, :] + depth_flat[valid, None] * ray_dirs[valid]
    return world_pts.cpu().numpy(), label_flat[valid].cpu().numpy()


def main(scene, checkpoint_dir, output_path, render_mode):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    from featurefoam_lerf_bridge import load_manifest_index

    wp.init()
    device = "cuda"
    config_path = f"{checkpoint_dir}/config.yaml"
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    args = parser.parse_args(["-c", config_path])
    data_handler = DataHandler(args)
    data_handler.reload("all", downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data_handler, device=device)
    model.load_pt(f"{checkpoint_dir}/model.pt")
    model.update_vis_cache()

    # VisOptions.depth_quantile has NO default in the renderer (warp struct, zero-initializes)
    # -- the default vis_options built inside Rasterizer.visualize() when none is passed only
    # sets transmittance_threshold/max_intersections/bkgd_color, leaving depth_quantile at 0.0,
    # which means `next_trans < options.depth_quantile` (transmittance monotonically decreasing
    # from 1.0) never fires and depth_out silently stays all-zero forever. Confirmed directly:
    # color/alpha/normal/intersections all render correctly with real values, only depth was
    # zero, isolating this exact cause rather than a broader rendering failure. Fixed by building
    # our own VisOptions with a real depth_quantile (0.5 = median-transmittance depth, a
    # reasonable single-value summary of "where the surface is" along each ray).
    from powerfoam.rasterize import VisOptions
    import warp as wp
    vis_options = VisOptions()
    vis_options.transmittance_threshold = 1e-3
    vis_options.max_intersections = 1024
    vis_options.depth_quantile = 0.5
    vis_options.bkgd_color = wp.vec3f(0.0, 0.0, 0.0)

    manifest_idx = load_manifest_index(scene)
    frames, cat_to_id = load_scene_labels(scene)
    print(f"{len(frames)} labeled frames, {len(cat_to_id)} categories: {list(cat_to_id.keys())}")

    all_points, all_labels = [], []
    for info in frames:
        frame_name = info["info"]["name"]
        if frame_name not in manifest_idx:
            print(f"  WARNING: {frame_name} not in manifest, skipping")
            continue
        cam_idx = manifest_idx[frame_name]
        camera = data_handler.cameras[cam_idx]

        result = model.forward_visualization(camera, render_mode=render_mode, vis_options=vis_options)
        depth = result[1]
        label_canvas = rasterize_frame_labels(info, cat_to_id, camera.width, camera.height)

        pts, lbl = unproject_labeled_frame(camera, depth, label_canvas, device)
        if pts is not None:
            all_points.append(pts)
            all_labels.append(lbl)
            print(f"  {frame_name}: {pts.shape[0]} labeled points unprojected")

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total: {points.shape[0]} points across {len(frames)} frames")
    print(f"bbox before outlier filtering: {points.min(0)} to {points.max(0)}")

    # A depth-quantile ray occasionally hits a stray floater primitive (see
    # powerfoam/scene.py::init_points_sfm's docstring / ResearchVault's floater writeup) instead
    # of the real intended surface, producing a spurious far-away 3D point for that pixel. The
    # 99.9th percentile of camera-relative distance is a tight, real cluster (matches this
    # scene's known ~3.5-6 unit radius from earlier camera/SfM analysis); anything past a
    # generous multiple of it is almost certainly such a stray hit, not real GT.
    dist_from_median = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    cutoff = max(10.0, 3.0 * np.percentile(dist_from_median, 99.9))
    keep = dist_from_median <= cutoff
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"Dropping {n_dropped}/{points.shape[0]} points beyond {cutoff:.2f} units "
              f"(3x the 99.9th percentile distance) as likely stray floater-primitive hits")
        points, labels = points[keep], labels[keep]
    print(f"bbox after outlier filtering: {points.min(0)} to {points.max(0)}")

    id_to_cat = {v: k for k, v in cat_to_id.items()}
    np.savez(output_path, points=points.astype(np.float32), labels=labels.astype(np.int64),
              class_id_to_name=json.dumps(id_to_cat))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--render-mode", default="rasterize", choices=["rasterize", "raytrace"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.scene, args.checkpoint_dir, args.output, args.render_mode)
