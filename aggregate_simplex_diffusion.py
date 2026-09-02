"""Aggregate run_simplex_diffusion_eval per-scene JSONs into paired 10-scene tables."""
import glob, json, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "artifacts/scannet/simplex_10scene"
files = sorted(f for f in glob.glob(os.path.join(d, "*.json"))
               if os.path.basename(f) != "summary.json")
res = [json.load(open(f)) for f in files]
scenes = [r["scene"] for r in res]
arms = list(res[0]["arms"].keys())
cs_list = list(res[0]["arms"]["base"].keys())
print(f"{len(scenes)} scenes: {', '.join(scenes)}\n")

out = {"scenes": scenes, "class_sets": {}}
for cs in cs_list:
    print(f"=== {cs} ===")
    base = np.array([r["arms"]["base"][cs]["mIoU"] for r in res])
    base_a = np.array([r["arms"]["base"][cs]["mAcc"] for r in res])
    print(f"{'arm':<28} {'mIoU':>7} {'d_base':>8} {'win':>5} {'mAcc':>7} {'dA':>7}")
    print(f"{'base':<28} {base.mean():7.2f} {'--':>8} {'--':>5} {base_a.mean():7.2f} {'--':>7}")
    rows = {}
    for arm in arms:
        if arm == "base":
            continue
        v = np.array([r["arms"][arm][cs]["mIoU"] for r in res])
        va = np.array([r["arms"][arm][cs]["mAcc"] for r in res])
        rows[arm] = {"mIoU": float(v.mean()), "mAcc": float(va.mean()),
                     "delta_base": float((v - base).mean()),
                     "wins_over_base": int((v > base).sum()),
                     "per_scene": {s: float(x) for s, x in zip(scenes, v)}}
        print(f"{arm:<28} {v.mean():7.2f} {(v-base).mean():+8.2f} "
              f"{int((v>base).sum()):>4}/{len(scenes)} {va.mean():7.2f} {(va-base_a).mean():+7.2f}")
    # paired true_facet vs cech at matched (s, a)
    print()
    for arm in arms:
        if not arm.startswith("true_facet_"):
            continue
        ce = arm.replace("true_facet_", "cech_", 1)
        if ce not in arms:
            continue
        v = np.array([r["arms"][arm][cs]["mIoU"] for r in res])
        w = np.array([r["arms"][ce][cs]["mIoU"] for r in res])
        rows.setdefault("_paired", {})[arm] = {
            "delta_tf_minus_cech": float((v - w).mean()),
            "std": float((v - w).std()), "wins": int((v > w).sum())}
        print(f"PAIRED {arm} - {ce}: {(v-w).mean():+.2f} "
              f"+/- {(v-w).std():.2f}  wins {int((v>w).sum())}/{len(scenes)}")
    out["class_sets"][cs] = {"base_mIoU": float(base.mean()),
                             "base_mAcc": float(base_a.mean()), "arms": rows}
    print()
json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=2)
print("wrote", os.path.join(d, "summary.json"))
