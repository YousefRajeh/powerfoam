"""Feature Foam <-> splat-distiller's LERFMetrics bridge.

splat-distiller's real evaluation pipeline (evaluator_loader.py's _eval + metrics.py's
LERFMetrics) works off files on disk: for each labeled frame it reads
<rendered_folder>/RGB/<frame>.{jpg,png} and <rendered_folder>/AttentionMap/<frame>.pt (an
(H, W, num_prompts) raw-dot-product attention-score tensor, prompts = dedup'd object
categories from that frame's json, IN FIRST-APPEARANCE ORDER, followed by the 8
BACKGROUND_WORDS from evaluator_loader.py). LERFMetrics.compute_metrics() itself does NOT
touch the renderer or the text encoder at all -- it only reads these files.

So the only thing that needs to be method-specific is producing those two files correctly.
This module renders them for Feature Foam (PowerFoam + solved primitive features), using
export_operator_for_views (the same sparse operator used throughout this project) and dots
attention scores BEFORE rendering -- matching GaussianRenderer's AttentionMap mode exactly
(raw, unnormalized primitive features; text features L2-normalized) -- then hands off
unchanged to their real LERFMetrics for the IoU/mAcc computation.

NOTE: the RGB file saved here is a copy of the ground-truth image, not a real PowerFoam
color render -- fine for mIoU/mAcc (only AttentionMap.pt drives those), but means any
PSNR/SSIM LERFMetrics also computes from this bridge is meaningless placeholder data, not a
real fidelity measurement. Flagged explicitly so it's never mistaken for a real PSNR/SSIM.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\splat-distiller")

LERF_ROOT = Path(r"D:\Downloads\powerfoam\data\lerf_ovs_raw\lerf_ovs")
BACKGROUND_WORDS = ["floor", "wall", "ceiling", "background", "object", "things", "stuff", "texture"]


def build_prompts_for_frame(label_json_path):
    info = json.loads(Path(label_json_path).read_text())
    categories = [obj["category"] for obj in info["objects"]]
    positives = list(dict.fromkeys(categories))  # first-appearance dedup, matches json_parser/_load_camera
    return positives, positives + BACKGROUND_WORDS


def load_manifest_index(scene, openclip_subdir="openclip_train"):
    manifest = json.loads(
        Path(f"D:/Downloads/powerfoam/artifacts/lerf_ovs/{scene}/{openclip_subdir}/feature_manifest.json").read_text()
    )
    return {Path(v["image"]).name: v["id"] for v in manifest["views"]}


def render_attention_map(scene, camera_index, primitive_features, text_features, model=None, data_handler=None):
    """text_features: (num_prompts, C) L2-normalized. primitive_features: (P, C) RAW
    (not normalized -- matches GaussianRenderer's AttentionMap mode, which dots the raw
    primitive feature against normalized text features with no primitive-side normalization).
    Returns (H, W, num_prompts) attention map for the given camera index, plus the
    model/data_handler (so callers can reuse them across frames instead of reloading)."""
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import export_operator_for_views

    if model is None:
        wp.init()
        config_path = f"D:/Downloads/powerfoam/output/lerf_ovs_{scene}/config.yaml"
        parser = configargparse.ArgParser()
        add_group(parser, Params)
        parser.add_argument("-c", "--config", is_config_file=True)
        args = parser.parse_args(["-c", config_path])
        data_handler = DataHandler(args)
        data_handler.reload("all", downsample=args.downsample[-1])
        model = PowerfoamScene(args)
        model.initialize_from_dataset(data_handler, device="cuda")
        model.load_pt(f"D:/Downloads/powerfoam/output/lerf_ovs_{scene}/model.pt")

    camera = data_handler.cameras[camera_index]
    a = export_operator_for_views(model, [camera], [camera_index])

    attention_scores_per_primitive = primitive_features @ text_features.T  # (P, num_prompts), matches AttentionMap mode
    rendered = a.matmul(attention_scores_per_primitive)  # (H*W, num_prompts)
    return rendered.reshape(camera.height, camera.width, -1), model, data_handler


def render_scene_for_lerf_eval(scene, result_folder, text_encoder_model="ViT-B-16", text_encoder_pretrained="laion2b_s34b_b88k",
                                solved_features_path=None):
    """Default text_encoder_model/pretrained MUST match splat-distiller/pre_processing.py's
    OpenCLIPNetworkConfig (clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k") --
    the model that actually produced the SAM+CLIP .npy features this bridge renders. Using a
    mismatched text encoder (e.g. "ViT-B-16-quickgelu"/"openai", Feature Foam's OWN dense-round
    convention) puts image and text embeddings in different, non-cross-compatible spaces: primitive
    self-similarity (e.g. click-to-segment in the viewer) stays coherent since it's within one
    embedding space, but text-query relevancy collapses to near-random since the text side is a
    different model entirely. Found the hard way: every SAM-round mIoU number this bridge produced
    while accidentally defaulted to "ViT-B-16-quickgelu"/"openai" collapsed to ~0.01-0.06 regardless
    of checkpoint/primitive count; re-rendering with the correct encoder reproduced the original
    0.5268 exactly. Confirm this default still matches pre_processing.py before changing it."""
    import open_clip

    result_folder = Path(result_folder)
    (result_folder / "RGB").mkdir(parents=True, exist_ok=True)
    (result_folder / "AttentionMap").mkdir(parents=True, exist_ok=True)

    clip_model, _, _ = open_clip.create_model_and_transforms(text_encoder_model, pretrained=text_encoder_pretrained)
    clip_model.eval().to("cuda")
    tokenizer = open_clip.get_tokenizer(text_encoder_model)

    manifest_idx = load_manifest_index(scene)
    if solved_features_path is None:
        solved_features_path = f"D:/Downloads/powerfoam/artifacts/lerf_ovs/{scene}/solved_weighted.pt"
    solved = torch.load(solved_features_path, map_location="cuda", weights_only=True)
    primitive_features = solved["primitive_features"].to("cuda").float()  # raw, NOT normalized -- see docstring

    label_dir = LERF_ROOT / "label" / scene
    model, data_handler = None, None
    for label_file in sorted(label_dir.glob("*.json")):
        info = json.loads(label_file.read_text())
        frame_name = info["info"]["name"]
        if frame_name not in manifest_idx:
            print(f"  WARNING: {frame_name} not in manifest, skipping")
            continue
        cam_idx = manifest_idx[frame_name]
        _, prompts = build_prompts_for_frame(label_file)

        with torch.no_grad():
            tok = tokenizer(prompts).to("cuda")
            text_features = clip_model.encode_text(tok).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        attn_map, model, data_handler = render_attention_map(scene, cam_idx, primitive_features, text_features, model, data_handler)

        basename = label_file.stem
        torch.save(attn_map.cpu(), result_folder / "AttentionMap" / f"{basename}.pt")
        gt_image_path = LERF_ROOT / scene / "images" / frame_name
        shutil.copyfile(gt_image_path, result_folder / "RGB" / f"{basename}{Path(frame_name).suffix}")
        print(f"  rendered {frame_name} -> {basename} ({len(prompts)} prompts)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--result-folder", required=True)
    p.add_argument("--text-encoder-model", default="ViT-B-16",
                    help="Must match splat-distiller/pre_processing.py's OpenCLIPNetworkConfig -- see render_scene_for_lerf_eval's docstring.")
    p.add_argument("--text-encoder-pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--solved-features", default=None)
    args = p.parse_args()
    render_scene_for_lerf_eval(args.scene, args.result_folder, args.text_encoder_model, args.text_encoder_pretrained,
                                solved_features_path=args.solved_features)
