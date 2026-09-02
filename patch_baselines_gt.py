"""Point run_baselines.py at the validated GT loader and class remapping.

`load_scannet_pointcept_gt(scene_dir, label_field)` takes a DIRECTORY under
D:\\Downloads\\scannet_pointcept\\{split}\\{scene} and returns (points, raw_labels, all_names).
The class subset is then formed by keeping the OpenGaussian names that are actually present in the
scene, and `remap_gt_labels` compacts them to 1..C with 0 = ignore -- the same sequence
backfill_surface_metrics.py and diagnose_lifting_rays.py use, so our rows are scored identically to
every other number in this project.
"""
P = "run_baselines.py"
s = open(P, encoding="utf-8").read()

s = s.replace(
    "SCENES = [", 'SPLIT = {"scene0645_00": "val"}      # every other scene is in train\nSCENES = [')

old = """    for scene in a.scenes.split(","):
        gt_pts, raw = load_scannet_pointcept_gt(scene)
        present = set(np.unique(raw).tolist())
        kept = [n for n in names if n in present] if isinstance(next(iter(present), 0), str) else names
        txt = embed_class_names(list(names), device)"""
new = '''    for scene in a.scenes.split(","):
        split = SPLIT.get(scene, "train")
        gt_pts, raw, names_all = load_scannet_pointcept_gt(
            rf"D:\\Downloads\\scannet_pointcept\\{split}\\{scene}", "segment20")
        n2i = {n: i for i, n in enumerate(names_all)}
        present = set(np.unique(raw).tolist())
        kept = [(n2i[n], n) for n in names if n in n2i and n2i[n] in present]
        if not kept:
            print(f"[skip] {scene}: no target classes present", flush=True)
            continue
        gt = remap_gt_labels(raw, [i for i, _ in kept])          # 0 = ignore, 1..C
        n_cls = len(kept) + 1
        txt = embed_class_names([n for _, n in kept], device)'''
assert old in s, "GT block not found"
s = s.replace(old, new)

# the per-method loop recomputed gt/n_classes from the wrong names list
s = s.replace("""                pred = lab[assigned]
                gt = remap_gt_labels(raw, list(range(len(names))))
                miou, macc = point_metrics(pred, gt, len(names) + 1)
                idx = GTSurfaceIndex(gt_pts, gt, len(names) + 1)""",
              """                pred = lab[assigned]
                miou, macc = point_metrics(pred, gt, n_cls)
                idx = GTSurfaceIndex(gt_pts, gt, n_cls)""")

open(P, "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print("run_baselines.py: GT loader + class remap corrected")
