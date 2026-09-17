# Pipeline work required for NormLift and LaGa surface metrics

Both baselines are blocked before the metric, not at it. Neither currently produces per-Gaussian
CLIP-space features, which is the one thing `run_baseline_surface.py` needs. Once a `(ckpt,
features)` pair exists the rest is mechanical — `make_baseline_eval_inputs.py` normalises it and the
surface/IoU runners take it from there, exactly as for the four baselines already done.

Everything below is what has to be built, in order, with the reason each step exists.

---

## NormLift

**Arm: frozen.** Its README states it trains "exactly as in OpenGaussian's Stage 0, i.e. with fixed
Gaussian positions and the densification process disabled" — same arm as LUDVIG and OpenGaussian.
**Level: 3 (large)**, per its hyperparameter table (`Lifting  Tikhonov / SAM level  1.0 / 3`), which
makes it the one baseline already level-matched to our `_ogl3` rows.

### Blocker 1 — data layout (the real work)

NormLift wants OpenGaussian layout, we have COLMAP + Pointcept:

| NormLift expects | We have | Action |
|---|---|---|
| `<scene>/transforms_train.json`, `transforms_test.json` | `sparse/0/{cameras,images}.bin` | **convert** |
| `<scene>/points3d.ply` | `D:\Downloads\scenes10_points3d\<scene>\points3d.ply` | symlink — already the labelled mesh |
| `<scene>/<scene>_vh_clean_2.labels.ply` | same file (it carries `face` + per-vertex labels) | symlink |
| `<scene>/language_features/language_features/*_{f,s}.npy` | `..._colmap/openclip_features_sam_blackboth` | symlink, note the **doubled** directory name |
| `<scene>_colmap/` | already exists | reuse |

The conversion is the only non-trivial piece. OpenGaussian ships `scripts/scannet2blender.py`, which
writes NeRF-style `transforms_*.json` from ScanNet poses; our COLMAP `images.bin` holds the same
poses in the other convention, so the safer route is a direct COLMAP -> transforms writer:

    R = qvec2rotmat(img.qvec); t = img.tvec          # world->camera
    c2w = inv([[R, t], [0,0,0,1]])
    c2w[:3, 1:3] *= -1                               # COLMAP (y down, z fwd) -> Blender/NeRF (y up, z back)
    frame = {"file_path": f"./color/{stem}", "transform_matrix": c2w.tolist()}
    out = {"camera_angle_x": 2*atan(W / (2*fx)), "frames": [...]}

Getting that sign flip wrong yields a scene that trains but reconstructs mirrored, and nothing
downstream would flag it — so **verify by re-rendering one view and comparing to the source image**
before running all 10.

NormLift's `preprocess/convert_to_colmap.py` goes the *other* direction (its data -> COLMAP), so it
does not help us; we already have the COLMAP side.

### Blocker 2 — external repos and deps

`SPLAT_DISTILLER_DIR` and `OPENGAUSSIAN_DIR` must point at real checkouts (both exist on 995), plus
`faiss` (installed) and `plyfile`. NormLift imports `gsplat_ext` from splat-distiller, so it inherits
the same `batch_ids` incompatibility that broke the pristine SFS clone — expect to point it at the
fork's `gsplat_ext`, whose only deltas are that unpack and an import move.

### Steps

1. `preprocess`: build `dataset/scannet/<scene>/` with the symlinks above + generated transforms.
2. `train/train_rgb.py --scene <scene>` — 30k iters, `-r 2`, frozen points.
   **Check first** whether our existing frozen gsplat checkpoint can be substituted; NormLift's own
   training is the same OpenGaussian Stage 0 recipe, so a conversion would save 10 x ~40 min.
3. `train/ply_to_pt.py --scene <scene>` — ply -> pt.
4. `NORMLIFT_SCENE=<scene> python lift/distill_features.py` — Tikhonov 1.0, SAM level 3.
5. `pipeline/run_pipeline.py --classes 19` — 5 steps/scene: visibility -> effective views ->
   confidence -> KNN vote -> eval. **Voting runs once in 19-class space**; `--classes 15/10
   --steps eval` re-scores the same products, so those are cheap.
6. Extract per-Gaussian features -> add a `("normlift", "frozen")` entry to
   `make_baseline_eval_inputs.py` -> `run_baseline_surface.py --tags normlift_frozen`.

**Watch:** NormLift's own contribution is confidence-weighted KNN *label* voting, so its final
product may be per-Gaussian **labels**, not 512-d features. If so, step 6 changes: the surface metric
needs predicted labels per GT point, which labels give directly — bypass
`make_baseline_eval_inputs.py` and feed `semantic_surface_metrics(points, gt, pred, ...)` straight,
skipping the CLIP argmax. Confirm which before writing the adapter.

---

## LaGa

**Arm: frozen**, matching the other ScanNet baselines. Contrastive features for all 10 scenes already
exist on 995 at
`~/LaGa/output/scannet-<scene>/point_cloud/iteration_30000/contrastive_feature_point_cloud.ply`.

### Blocker — the CLIP stage is not in their shipped scripts

`get_scannet_features.sh` stops at `train_affinity_features.py`, which is what produced those plys.
Those are **contrastive** embeddings, not CLIP: they support grouping, not text queries. The step
that turns groups into text-comparable features lives in their GUI/eval path
(`laga_gui.py`, `metrics.py`), not in a runnable script.

### Steps

1. Read `laga_gui.py` / `metrics.py` for the query path and find where a text embedding is compared
   to something derived from the contrastive features. That defines the missing stage; likely:
   cluster contrastive features -> for each cluster gather its 2D masks -> pool CLIP embeddings of
   those masks -> assign the pooled CLIP vector to every Gaussian in the cluster.
2. Reimplement that as a script over the 10 scenes, reusing the SAM+CLIP features we already have
   (`openclip_features_sam_blackboth`) rather than re-extracting.
3. Output `(N, 512)` per-Gaussian CLIP features aligned to `scene_point_cloud.ply`'s Gaussian order
   — the contrastive ply and the scene ply must be index-aligned; **verify counts match** before
   trusting the alignment.
4. Add `("laga", "frozen")` to the adapter and run the two evaluators.

**Cost note:** step 2 is a genuine reimplementation of an unshipped stage, so LaGa is the most
expensive of the outstanding baselines and the one most likely to differ from their published
numbers. Worth stating in the paper if its row lands materially below the published 32.50.

---

## Not doing

**THGS** — needs 2DGS reconstructions (its `gaussian_model.py` concatenates a ones-column to a
2-component `_scaling`, the surfel convention), so our 3DGS checkpoints cannot load. That is a whole
reconstruction stage for 10 scenes, not a bridge.
