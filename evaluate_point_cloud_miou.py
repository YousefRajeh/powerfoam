"""OpenGaussian-style 3D point-level mIoU/mAcc evaluation (github.com/yanmin-wu/OpenGaussian,
scripts/eval_scannet.py), applied to our own back-projected GT (unproject_replica_gt.py) instead
of ScanNet's official labeled mesh, and to both PowerFoam and the Splat Feature Solver (3DGS)
baseline via each method's own natural point<->primitive correspondence
(point_cloud_query.py: exact power-cell membership for PowerFoam, nearest-Gaussian-center for
the 3DGS baseline -- see that module's docstring for why they differ).

`calculate_metrics` below is a direct port of OpenGaussian's own function (verified verbatim via
curl against their repo) -- same per-scene-present-classes-only mIoU/mAcc, same 0=ignore
convention -- not a reimplementation of the scoring logic, only the correspondence step differs
from their ScanNet-specific mechanism.
"""
import argparse
import json
from pathlib import Path

from determinism import enable_determinism
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from point_cloud_query import assign_points_to_power_cells, assign_points_to_nearest_center

CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"  # must match splat-distiller/pre_processing.py's
                                        # OpenCLIPNetworkConfig -- the checkpoint that produced
                                        # every solved feature field this script consumes.


def calculate_metrics(gt, pred, total_classes):
    """Ported verbatim (logic-for-logic) from OpenGaussian's scripts/eval_scannet.py."""
    gt = gt.cpu()
    pred = pred.cpu()
    pred = pred.clone()
    pred[gt == 0] = 0

    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)
    ious = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
        correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        total[cls] = torch.sum(gt == cls).item()

    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]

    mean_iou = ious[valid_gt_classes].mean().item()

    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else float("nan")

    class_accuracy = torch.where(total > 0, correct / total.clamp_min(1), torch.zeros_like(total))
    mean_class_accuracy = class_accuracy[valid_gt_classes].mean().item()

    return ious, mean_iou, accuracy, mean_class_accuracy


def remap_gt_labels(raw_labels, target_ids):
    """target_ids: list of raw class ids to keep, in the order they'll be assigned contiguous
    1..K labels (0 = ignore/other, same convention as OpenGaussian)."""
    id_to_new = {raw_id: i + 1 for i, raw_id in enumerate(target_ids)}
    remapped = np.zeros_like(raw_labels)
    for raw_id, new_id in id_to_new.items():
        remapped[raw_labels == raw_id] = new_id
    return remapped


# OpenGaussian's shipped text_features.json (assets/text_features.zip) tokenizes this class
# as ONE word, and LUDVIG loads that exact file, so every ScanNet baseline queries
# "showercurtain". Verified by comparing embeddings: 18 of our 19 class vectors match theirs
# to cos = 1.0000, and this one sat at 0.8621 purely because of the space. Rewriting it here
# makes our text bank byte-identical to the baselines' rather than differing on one class.
OPENGAUSSIAN_NAME_OVERRIDES = {"shower curtain": "showercurtain"}


def embed_class_names(class_names, device, match_opengaussian=True):
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    if match_opengaussian:
        class_names = [OPENGAUSSIAN_NAME_OVERRIDES.get(n, n) for n in class_names]
    with torch.no_grad():
        tok = tokenizer(class_names).to(device)
        text_feats = clip_model.encode_text(tok).float()
    return F.normalize(text_feats, dim=-1)


def classify_primitives(primitive_features, text_feats, hubness_correct=True):
    """(P, C) raw primitive features x (K, C) normalized text features -> (P,) argmax class
    index in [0, K-1] (caller adds +1 to land in the 1..K GT label space).

    `hubness_correct`: per-class z-score standardization of the similarity matrix before
    argmax (CSLS-style hubness correction, e.g. Lample et al. 2018 "Word Translation without
    Parallel Data") -- diagnosed as necessary here: some class text embeddings (e.g. "floor")
    have high but low-VARIANCE cosine similarity to nearly everything, while others (e.g.
    "window") have a lower mean but occasionally spike higher, so plain per-primitive argmax
    systematically favors the high-variance class regardless of true relevance (confirmed via
    the per-class mean/std similarity breakdown: floor mean=0.216/std=0.013 vs window
    mean=0.216/std=0.028, equal means but window wins far more argmax comparisons). Standardizing
    each class's similarity column to zero-mean/unit-variance before comparing removes this
    scale bias. This is a real, literature-grounded correction for a documented phenomenon, not
    a tuned heuristic -- but even after it, one class can still noticeably over-predict (see
    ResearchVault writeup), so treat these numbers as a first, imperfect pass, not a final one.
    """
    unit_features = F.normalize(primitive_features, dim=-1)
    sim = unit_features @ text_feats.T  # (P, K)
    if hubness_correct:
        sim = (sim - sim.mean(dim=0, keepdim=True)) / sim.std(dim=0, keepdim=True).clamp_min(1e-6)
    return sim.argmax(dim=-1)


def evaluate_powerfoam(gt_points, gt_labels_remapped, num_classes, target_names, device,
                        checkpoint_dir, solved_features_path):
    import warp as wp
    import configargparse
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene

    wp.init()
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

    centers = model.points.detach().cpu().numpy()
    radii = model.get_radii().detach().cpu().numpy()

    solved = torch.load(solved_features_path, map_location=device, weights_only=True)
    primitive_features = solved["primitive_features"].to(device).float()
    valid_mask = solved["valid_mask"].cpu().numpy()
    print(f"[powerfoam] {centers.shape[0]} primitives, {valid_mask.sum()} valid (support>0)")

    text_feats = embed_class_names(target_names, device)
    primitive_class = classify_primitives(primitive_features, text_feats).cpu().numpy()

    assigned = assign_points_to_power_cells(gt_points, centers, radii, valid=valid_mask, k=64)
    pred_labels = np.zeros(gt_points.shape[0], dtype=np.int64)
    owned = assigned >= 0
    pred_labels[owned] = primitive_class[assigned[owned]] + 1  # -> 1..K label space

    gt_t = torch.from_numpy(gt_labels_remapped).long()
    pred_t = torch.from_numpy(pred_labels).long()
    return calculate_metrics(gt_t, pred_t, num_classes + 1)


def load_gaussian_means_opacities(gaussian_ckpt_path, device):
    """Two supported checkpoint formats: splat-distiller's native `.pt` ({"splats": {...}},
    pre-activation opacity) and a graphdeco-style `.ply` (e.g. from
    convert_opengaussian_ckpt_to_ply.py -- same pre-activation opacity convention, just stored
    as a PLY vertex field instead of a torch tensor)."""
    if str(gaussian_ckpt_path).lower().endswith(".ply"):
        from plyfile import PlyData
        ply = PlyData.read(gaussian_ckpt_path)
        v = ply["vertex"].data
        means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        opacities = torch.sigmoid(torch.from_numpy(np.array(v["opacity"], dtype=np.float32))).numpy()
    else:
        ckpt = torch.load(gaussian_ckpt_path, map_location=device, weights_only=False)["splats"]
        means = ckpt["means"].detach().cpu().numpy()
        opacities = torch.sigmoid(ckpt["opacities"]).detach().cpu().numpy()
    return means, opacities


def evaluate_splat_feature_solver(gt_points, gt_labels_remapped, num_classes, target_names, device,
                                   gaussian_ckpt_path, solved_features_path, opacity_threshold=0.1):
    means, opacities = load_gaussian_means_opacities(gaussian_ckpt_path, device)
    valid_mask = opacities >= opacity_threshold
    print(f"[splat-feature-solver] {means.shape[0]} gaussians, {valid_mask.sum()} valid (opacity>={opacity_threshold})")

    primitive_features = torch.load(solved_features_path, map_location=device, weights_only=False).float()
    assert primitive_features.shape[0] == means.shape[0]

    text_feats = embed_class_names(target_names, device)
    primitive_class = classify_primitives(primitive_features, text_feats).cpu().numpy()

    assigned = assign_points_to_nearest_center(gt_points, means, valid=valid_mask)
    pred_labels = np.zeros(gt_points.shape[0], dtype=np.int64)
    owned = assigned >= 0
    pred_labels[owned] = primitive_class[assigned[owned]] + 1

    gt_t = torch.from_numpy(gt_labels_remapped).long()
    pred_t = torch.from_numpy(pred_labels).long()
    return calculate_metrics(gt_t, pred_t, num_classes + 1)


# Coarse, common, structurally-significant classes only -- mirroring OpenGaussian's own ScanNet
# evaluation scope (their 19/15/10-class NYU40 subsets: wall, floor, cabinet, bed, chair, sofa,
# table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, shower curtain,
# toilet, sink, bathtub -- big furniture/architecture, no rare or ambiguous small objects).
# Diagnosed via the predicted-class histogram: bare 28-class argmax over EVERY Replica label
# (including "blinds", "picture", "wall-plug", "plant-stand", etc.) let two visually-unrelated
# words ("picture", "blinds") dominate over 60% of all primitive predictions regardless of true
# content -- a known CLIP zero-shot argmax pitfall (some text embeddings are generic
# "attractors"), made worse (not better) by adding a "a photo of a {class}" prompt template
# (tested: "picture" share rose to 45%), so the fix is narrowing the vocabulary to the kind of
# coarse, common, low-ambiguity classes OpenGaussian itself evaluates on, not a prompting tweak.
COARSE_CLASS_NAMES = [
    "wall", "floor", "ceiling", "chair", "sofa", "table", "window", "door", "cabinet",
    "lamp", "rug", "cushion",
]

# The standard ScanNet 20-class benchmark label set (used by every ScanNet semantic
# segmentation paper, including OpenGaussian's own 19/20-class NYU40 subset -- this ordering
# matches Pointcept's `segment20.npy` label ids 0-19 directly, -1 = ignore/unlabeled).
SCANNET20_CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
]

# OpenGaussian's OWN three class subsets, exactly as defined in their scripts/eval_scannet.py
# (verified verbatim via curl against their repo, not inferred): target_id lists into the NYU40
# taxonomy, translated here into names within SCANNET20_CLASS_NAMES's own ordering/spelling.
# 19-class: NYU40 ids [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36] -- this is EXACTLY
# SCANNET20_CLASS_NAMES minus "otherfurniture" (id 39, excluded from their 19-class set).
OPENGAUSSIAN_19_CLASS_NAMES = SCANNET20_CLASS_NAMES[:19]
# 15-class: NYU40 ids [1,2,3,4,5,6,7,8,9,10,12,14,16,33,34] -- drops picture, refrigerator,
# shower curtain, bathtub, otherfurniture relative to the 19-class set.
OPENGAUSSIAN_15_CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "counter", "desk", "curtain", "toilet", "sink",
]
# 10-class: NYU40 ids [1,2,4,5,6,7,8,9,10,33] -- drops cabinet, counter, desk, curtain,
# refrigerator, shower curtain, sink, bathtub, otherfurniture relative to the 19-class set.
OPENGAUSSIAN_10_CLASS_NAMES = [
    "wall", "floor", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "toilet",
]
OPENGAUSSIAN_CLASS_SETS = {
    "opengaussian19": OPENGAUSSIAN_19_CLASS_NAMES,
    "opengaussian15": OPENGAUSSIAN_15_CLASS_NAMES,
    "opengaussian10": OPENGAUSSIAN_10_CLASS_NAMES,
}


def load_scannet_pointcept_gt(scene_dir, label_field="segment20"):
    """Load ScanNet's real GT from Pointcept's per-scene .npy format
    (coord.npy + segment20.npy/segment200.npy, ids aligned 1:1 by row). Unlike
    unproject_replica_gt.py/unproject_lerf_gt.py, this is an OFFICIAL point-level GT (matches
    OpenGaussian's own ScanNet protocol exactly), not something we built ourselves --
    confirmed: this coord.npy's point count matches both the official
    `{scene}_vh_clean_2.labels.ply` mesh vertex count AND the OpenGaussian-trained Gaussian
    checkpoint's own point count exactly (81,369 for scene0000_00), proving the Gaussian
    reconstruction used frozen initialization from this same point set.
    """
    scene_dir = Path(scene_dir)
    points = np.load(scene_dir / "coord.npy").astype(np.float32)
    raw_labels = np.load(scene_dir / f"{label_field}.npy").astype(np.int64)
    # segment20 ids are already 0..19 with -1=ignore; target_ids=[0..19] below maps them to the
    # pipeline's 1..K convention (0=ignore) via remap_gt_labels, same as any other GT source.
    return points, raw_labels, SCANNET20_CLASS_NAMES


def main(args):
    enable_determinism()   # bitwise-reproducible eval; see determinism.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.gt_format == "scannet":
        gt_points, raw_labels, all_names = load_scannet_pointcept_gt(args.gt_points, args.scannet_label_field)
        num_all_classes = len(all_names)  # 20 for segment20, 200 for segment200
        if args.classes == "all":
            target_ids = list(range(num_all_classes))
        else:
            if args.classes == "coarse":
                wanted_names = COARSE_CLASS_NAMES
            elif args.classes in OPENGAUSSIAN_CLASS_SETS:
                wanted_names = OPENGAUSSIAN_CLASS_SETS[args.classes]
            else:
                wanted_names = args.classes.split(",")
            name_to_id = {n: i for i, n in enumerate(all_names)}
            target_ids = [name_to_id[n] for n in wanted_names if n in name_to_id]
            missing = [n for n in wanted_names if n not in name_to_id]
            if missing:
                print(f"WARNING: classes not in {args.scannet_label_field}'s vocabulary, skipped: {missing}")
        target_names = [all_names[i] for i in target_ids]
    else:
        gt_data = np.load(args.gt_points, allow_pickle=True)
        gt_points = gt_data["points"]
        raw_labels = gt_data["labels"]
        class_id_to_name = json.loads(str(gt_data["class_id_to_name"]))
        name_to_id = {v: int(k) for k, v in class_id_to_name.items()}

        if args.classes == "all":
            target_ids = sorted(set(raw_labels.tolist()))
        else:
            wanted_names = COARSE_CLASS_NAMES if args.classes == "coarse" else args.classes.split(",")
            target_ids = [name_to_id[n] for n in wanted_names if n in name_to_id]
            missing = [n for n in wanted_names if n not in name_to_id]
            if missing:
                print(f"WARNING: classes not present in this scene's GT, skipped: {missing}")
        target_names = [class_id_to_name[str(i)] for i in target_ids]

    # OpenGaussian's own convention (eval_scannet.py): average mIoU/mAcc only over classes
    # actually PRESENT in this scene's GT, not just present in the overall vocabulary. Missing
    # this caused a real bug caught by inspection: scene0062_00 (a bathroom) has zero
    # chair/sofa/table/window/cabinet points at all, so those classes trivially scored IoU=0.0
    # (no possible true positives), dragging the "mean" down over classes that were never a fair
    # test to begin with. Filter here, uniformly for both GT formats.
    present_ids = set(np.unique(raw_labels).tolist())
    absent_in_scene = [n for i, n in zip(target_ids, target_names) if i not in present_ids]
    if absent_in_scene:
        print(f"NOTE: classes in target vocabulary but absent from this scene's GT (excluded "
              f"from the per-scene mean, per OpenGaussian's own convention): {absent_in_scene}")
    kept = [(i, n) for i, n in zip(target_ids, target_names) if i in present_ids]
    target_ids = [i for i, _ in kept]
    target_names = [n for _, n in kept]

    print(f"Evaluating over {len(target_ids)} classes: {list(zip(target_ids, target_names))}")

    gt_labels_remapped = remap_gt_labels(raw_labels, target_ids)
    num_classes = len(target_ids)

    results = {}

    if args.method in ("powerfoam", "both"):
        ious, miou, acc, macc = evaluate_powerfoam(
            gt_points, gt_labels_remapped, num_classes, target_names, device,
            args.powerfoam_checkpoint, args.powerfoam_features,
        )
        print(f"\n[PowerFoam] mIoU={miou:.4f} mAcc={macc:.4f} overall_acc={acc:.4f}")
        for i, name in enumerate(target_names):
            print(f"    {name}: IoU={ious[i+1]:.4f}")
        results["powerfoam"] = {
            "mIoU": miou, "mAcc": macc, "overall_acc": acc,
            "per_class_iou": {name: float(ious[i + 1]) for i, name in enumerate(target_names)},
        }

    if args.method in ("splat_feature_solver", "both"):
        ious, miou, acc, macc = evaluate_splat_feature_solver(
            gt_points, gt_labels_remapped, num_classes, target_names, device,
            args.gaussian_checkpoint, args.gaussian_features,
        )
        print(f"\n[Splat Feature Solver] mIoU={miou:.4f} mAcc={macc:.4f} overall_acc={acc:.4f}")
        for i, name in enumerate(target_names):
            print(f"    {name}: IoU={ious[i+1]:.4f}")
        results["splat_feature_solver"] = {
            "mIoU": miou, "mAcc": macc, "overall_acc": acc,
            "per_class_iou": {name: float(ious[i + 1]) for i, name in enumerate(target_names)},
        }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt-points", default=r"D:\Downloads\powerfoam\artifacts\replica_room0\gt_point_cloud_3d.npz")
    p.add_argument("--gt-format", choices=["npz", "scannet"], default="npz",
                    help="'npz' (default): room_0/LERF-OVS back-projected GT (unproject_*_gt.py "
                    "output). 'scannet': Pointcept's per-scene .npy GT -- --gt-points must then be "
                    "the scene's directory (containing coord.npy/segment20.npy/etc.), not a file.")
    p.add_argument("--scannet-label-field", default="segment20",
                    help="Which Pointcept label file to use when --gt-format scannet: 'segment20' "
                    "(official 20-class ScanNet benchmark labels, default) or 'segment200'.")
    p.add_argument("--method", choices=["powerfoam", "splat_feature_solver", "both"], default="both")
    p.add_argument("--classes", default="coarse",
                    help="'coarse' (default, see COARSE_CLASS_NAMES), 'opengaussian19'/'opengaussian15'/"
                    "'opengaussian10' (OpenGaussian's own exact NYU40 class subsets, for a literal "
                    "comparison against their reported paper numbers), 'all' (every class present in "
                    "the GT -- diagnosed as too fine-grained/ambiguous for direct CLIP argmax, kept "
                    "only for reference), or a comma-separated custom list of class names.")
    p.add_argument("--output-json", default=None, help="Write results (mIoU/mAcc/per-class IoU) to this path for aggregation across scenes.")
    p.add_argument("--powerfoam-checkpoint", default=r"D:\Downloads\powerfoam\output\room_0")
    p.add_argument("--powerfoam-features", default=r"D:\Downloads\powerfoam\artifacts\replica_room0\solved_geometric_median_sam_v2.pt")
    p.add_argument("--gaussian-checkpoint", default=r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\ckpts\ckpt_29999_rank0.pt")
    p.add_argument("--gaussian-features", default=r"D:\Downloads\powerfoam\artifacts\room0_splatdistiller\ckpts\ckpt_29999_rank0_features.pt")
    main(p.parse_args())
