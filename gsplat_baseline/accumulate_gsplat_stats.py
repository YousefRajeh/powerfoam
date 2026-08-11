"""Stream-accumulate Feature Foam per-primitive statistics for the trained
gsplat garden checkpoint, across its 161 train views -- the Gaussian-splat
analogue of PowerFoam's accumulate_feature_stats.py, using the SAME OpenCLIP
feature archive (garden's 2D CLIP features don't depend on which 3D
representation is being lifted onto).
"""
import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gsplat_operator import export_view_operator  # noqa: E402

from feature_foam_lifting.operator import AccumulatedFeatureStats, normalize_features  # noqa: E402

CKPT_DIR = os.environ.get("GSPLAT_ARTIFACT_DIR", r"D:\Downloads\powerfoam\artifacts\garden_gsplat")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")
FEATURE_MANIFEST = r"D:\Downloads\powerfoam\artifacts\garden\openclip_train_all\feature_manifest.json"
OUTPUT_PATH = os.path.join(CKPT_DIR, "train_stats_161views.pt")
DEVICE = "cuda"


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    means, quats, scales, opacities, colors = (
        ckpt["means"].to(DEVICE), ckpt["quats"].to(DEVICE),
        torch.exp(ckpt["scales"]).to(DEVICE), torch.sigmoid(ckpt["opacities"]).to(DEVICE),
        torch.sigmoid(ckpt["colors"]).to(DEVICE),
    )
    K, width, height = ckpt["K"].to(DEVICE), ckpt["width"], ckpt["height"]
    camtoworlds = ckpt["camtoworlds"]
    viewmats_all = torch.linalg.inv(camtoworlds).to(DEVICE)
    train_idx = ckpt["train_idx"]
    num_primitives = means.shape[0]
    print(f"[accumulate_gsplat_stats] {num_primitives} gaussians, {len(train_idx)} train views, {width}x{height}")

    manifest = json.loads(Path(FEATURE_MANIFEST).read_text())
    manifest_dir = Path(FEATURE_MANIFEST).parent
    views_by_id = {v["id"]: v for v in manifest["views"]}
    print(f"[accumulate_gsplat_stats] loading combined feature archive ({manifest['feature_archive']})...")
    feature_maps_cpu = torch.load(manifest_dir / manifest["feature_archive"], map_location="cpu", weights_only=True)

    # The manifest's own "id" is a TRAIN-SPLIT-LOCAL sequential index (0..160,
    # in filename-sorted order among train images only) -- NOT the global
    # sorted-over-all-185-images index this script otherwise uses everywhere
    # else (matching ckpt["train_idx"]/camtoworlds). Map by filename, not by
    # arithmetic on the index, to avoid an off-by-one: an earlier version of
    # this script indexed the manifest directly by the global id, which
    # silently pulled the WRONG image's CLIP features for every view (the
    # lookup only ever failed once a global id exceeded the manifest's
    # max id of 160, well after ~140 views had already been silently
    # misaligned).
    manifest_id_by_filename = {Path(v["image"]).name: v["id"] for v in manifest["views"]}
    image_names = ckpt["image_names"]

    def local_manifest_id(global_view_id):
        filename = Path(image_names[global_view_id]).name
        if filename not in manifest_id_by_filename:
            raise SystemExit(f"feature manifest has no entry for {filename} (global view {global_view_id})")
        return manifest_id_by_filename[filename]

    def load_feature_map(global_view_id):
        local_id = local_manifest_id(global_view_id)
        grid = feature_maps_cpu[local_id].to(DEVICE).float()  # (grid_h, grid_w, C), raw
        record = views_by_id[local_id]
        upsampled = F.interpolate(
            grid.permute(2, 0, 1)[None], size=(int(record["height"]), int(record["width"])),
            mode="bilinear", align_corners=False,
        )[0]
        return normalize_features(upsampled.permute(1, 2, 0))

    feature_dim = load_feature_map(int(train_idx[0])).shape[-1]
    # CPU accumulators: at 3.64M gaussians x 512-d CLIP features, each (P, F)
    # accumulator is ~6.94GB, and accumulate_view briefly holds several such
    # tensors live at once (numerator, sq_numerator, gm_z, plus its own
    # per-view temporaries) -- enough to OOM a 48GB GPU that's also holding
    # the 3.64M-gaussian model itself. This machine has 274GB of system RAM,
    # so accumulating on CPU (only the per-view export/gather runs on GPU)
    # trades a few minutes of extra wall-clock for headroom that comfortably
    # fits, run once across 161 views rather than in a tight training loop.
    stats = AccumulatedFeatureStats.zeros(num_primitives=num_primitives, feature_dim=feature_dim, device="cpu")
    stats.positions = means.detach().cpu()

    t0 = time.time()
    for i, view_id in enumerate(train_idx.tolist()):
        row_indices, col_indices, values, _, _ = export_view_operator(
            means, quats, scales, opacities, colors, viewmats_all[view_id], K, width, height,
        )
        # Gather on CPU too: a (nnz, 512) gather at gsplat's ~34 hits/pixel
        # (vs PowerFoam's ~7) is ~19GB for a single view, on top of whatever
        # else is already resident on GPU -- move row_indices/feature_map to
        # CPU first so the gather itself never touches VRAM.
        feature_map = load_feature_map(view_id).cpu()  # (H, W, C)
        b = feature_map.view(-1, feature_dim)[row_indices.cpu()]
        stats.accumulate_view(col_indices.cpu(), values.cpu(), b)
        if i % 20 == 0:
            print(f"[accumulate_gsplat_stats] folded in {i + 1}/{len(train_idx)} views, nnz_this_view={row_indices.numel()}")

    elapsed = time.time() - t0
    diag = stats.diagnostics()
    print(f"[accumulate_gsplat_stats] TIMING {len(train_idx)} views elapsed_sec={elapsed:.3f} "
          f"valid_fraction={diag['valid_fraction']:.4f} reliability_mean={diag['reliability_mean']:.4f}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    stats.save(OUTPUT_PATH)
    print(f"[accumulate_gsplat_stats] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
