#!/usr/bin/env bash
# Lift the OpenGaussian-protocol L3 CLIP features onto every radfoam reconstruction.
#
# WHY REMOTE. The radfoam CUDA extension cannot be imported on the Windows box: the only
# local build (D:\Downloads\foamvol) is the cryo-ET fork, which exposes create_ct_pipeline
# instead of create_pipeline, so RadFoamScene cannot be constructed there. The remote already
# has upstream radfoam working and holds every radfoam checkpoint natively.
#
# --save-stats is passed so the SOLVER AXIS of the ablation (ridge / inverse-variance /
# consensus) can be filled in later without re-lifting; without it the per-view statistics are
# discarded after the geometric-median solve and every other solver becomes unrunnable.
#
# --sam-level 0 is correct and is NOT a typo for 3: the single-level artifact produced by
# SAM_ONLY_LEVEL=l stores its one granularity (level 3, whole-object) at index 0.
set +u
cd ~/radfoam || exit 1
Q=~/radfoam_lift
mkdir -p $Q/done $Q/logs

# Hardest-first, matching the extraction order.
SCENES="scene0062_00 scene0347_00 scene0070_00 scene0140_00 scene0645_00 scene0590_00 scene0000_00 scene0097_00 scene0200_00 scene0400_00"

for s in $SCENES; do
  FEAT=~/powerfoam/data/scannet/${s}_colmap/openclip_features_sam_l3
  NIMG=$(ls ~/powerfoam/data/scannet/${s}_colmap/images 2>/dev/null | wc -l)
  NEED=$((2 * NIMG))
  for arm in rf_match rf_unfroz; do
    tag="${arm}_${s}"
    [ -f "$Q/done/$tag" ] && { echo "[skip] $tag"; continue; }
    CKPT=~/radfoam/output/${arm}_${s}/model.pt
    [ -f "$CKPT" ] || { echo "[miss] $tag: no checkpoint"; continue; }
    HAVE=$(ls $FEAT 2>/dev/null | wc -l)
    if [ "$HAVE" -lt "$NEED" ]; then echo "[wait] $tag: features $HAVE/$NEED"; continue; fi
    OUT=~/radfoam/artifacts/scannet/${s}
    mkdir -p "$OUT"
    echo "[lift] $tag $(date +%H:%M:%S)"
    t0=$(date +%s)
    python scripts/lift_clip_features.py \
      --colmap-dir ~/powerfoam/data/scannet/${s}_colmap \
      --checkpoint "$CKPT" \
      --feature-folder "$FEAT" \
      --output "$OUT/solved_gm_${arm}_ogl3.pt" \
      --save-stats "$OUT/stats_${arm}_ogl3.pt" \
      --sam-level 0 --device cuda > "$Q/logs/${tag}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then touch "$Q/done/$tag"; echo "  ok ${dt}s"; else echo "  FAILED rc=$rc (${dt}s)"; tail -5 "$Q/logs/${tag}.log"; fi
  done
done
echo "[LIFT PASS DONE] $(date +%H:%M:%S)"
