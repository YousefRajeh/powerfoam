"""Same as featurefoam_lerf_bridge.py::render_scene_for_lerf_eval, but with an explicit
checkpoint directory override -- needed when the scene's normal output/lerf_ovs_{scene}/
directory is mid-retrain and we want to render against a backed-up prior checkpoint instead."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, r"D:\Downloads\powerfoam")
sys.path.insert(0, r"D:\Downloads\claude_logs")

from featurefoam_lerf_bridge import build_prompts_for_frame, load_manifest_index, LERF_ROOT


def render_attention_map_custom(checkpoint_dir, camera_index, primitive_features, text_features, model=None, data_handler=None):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    from powerfoam.feature_operator import export_operator_for_views

    if model is None:
        wp.init()
        config_path = f"{checkpoint_dir}/config.yaml"
        parser = configargparse.ArgParser()
        add_group(parser, Params)
        parser.add_argument("-c", "--config", is_config_file=True)
        args = parser.parse_args(["-c", config_path])
        data_handler = DataHandler(args)
        data_handler.reload("all", downsample=args.downsample[-1])
        model = PowerfoamScene(args)
        model.initialize_from_dataset(data_handler, device="cuda")
        model.load_pt(f"{checkpoint_dir}/model.pt")

    camera = data_handler.cameras[camera_index]
    a = export_operator_for_views(model, [camera], [camera_index])
    attention_scores_per_primitive = primitive_features @ text_features.T
    rendered = a.matmul(attention_scores_per_primitive)
    return rendered.reshape(camera.height, camera.width, -1), model, data_handler


def render_scene_for_lerf_eval_custom(scene, checkpoint_dir, result_folder, solved_features_path,
                                       text_encoder_model="ViT-B-16", text_encoder_pretrained="laion2b_s34b_b88k"):
    # Must match splat-distiller/pre_processing.py's OpenCLIPNetworkConfig -- see
    # featurefoam_lerf_bridge.py::render_scene_for_lerf_eval's docstring for the full story
    # (a mismatched default here silently collapsed every SAM-round mIoU this session).
    import open_clip

    result_folder = Path(result_folder)
    (result_folder / "RGB").mkdir(parents=True, exist_ok=True)
    (result_folder / "AttentionMap").mkdir(parents=True, exist_ok=True)

    clip_model, _, _ = open_clip.create_model_and_transforms(text_encoder_model, pretrained=text_encoder_pretrained)
    clip_model.eval().to("cuda")
    tokenizer = open_clip.get_tokenizer(text_encoder_model)

    manifest_idx = load_manifest_index(scene)
    solved = torch.load(solved_features_path, map_location="cuda", weights_only=True)
    primitive_features = solved["primitive_features"].to("cuda").float()

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

        attn_map, model, data_handler = render_attention_map_custom(checkpoint_dir, cam_idx, primitive_features, text_features, model, data_handler)

        basename = label_file.stem
        torch.save(attn_map.cpu(), result_folder / "AttentionMap" / f"{basename}.pt")
        gt_image_path = LERF_ROOT / scene / "images" / frame_name
        shutil.copyfile(gt_image_path, result_folder / "RGB" / f"{basename}{Path(frame_name).suffix}")
        print(f"  rendered {frame_name} -> {basename} ({len(prompts)} prompts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--result-folder", required=True)
    p.add_argument("--solved-features", required=True)
    args = p.parse_args()
    render_scene_for_lerf_eval_custom(args.scene, args.checkpoint_dir, args.result_folder, args.solved_features)
