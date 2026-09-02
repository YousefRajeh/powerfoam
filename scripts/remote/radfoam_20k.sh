#!/usr/bin/env bash
# Reconstruct all 10 ScanNet scenes, frozen + unfrozen, on RADFOAM'S OWN 20k SCHEDULE.
#
# WHY. Our existing radfoam runs used 30k iterations to match PowerFoam. Radiant Foam's shipped
# config (configs/mipnerf360_indoor.yaml) uses iterations 20_000 with freeze_points 18_000 and
# densify 2_000-11_000. The extra 10k iterations are the leading suspect for the measured
# pathology: on scene0062_00, 32.9% of rf_froz sites end at density < 1e-6, and those vacuum
# cells still own Voronoi territory containing 36% of GT points, which is where the 12-point
# mIoU deficit comes from. The photometric loss constrains the PATH INTEGRAL of density along
# each ray, not per-cell density, so a longer schedule gives the optimizer more opportunity to
# sparsify density into whichever subset minimises train loss -- free for PSNR, fatal for a
# partition-based semantic evaluation.
#
# VERIFIED FIRST (the reason this is a config experiment and not a bug hunt): our radfoam diff
# vs upstream is +131/-0 in pipeline.cu, +23/-0 in pipeline.h, +128/-0 in pipeline_bindings.cpp
# -- purely additive trace_export_weights for CLIP lifting, touching neither trace_forward nor
# trace_backward. train.py has exactly one deleted line, gating densification on
# `not frozen_init_pts`, which for the UNFROZEN arm is logically identical to upstream. The
# 30k reconstructions ran stock radfoam.
#
# Outputs go to rf20k_* so the 30k runs stay intact and the two can be compared directly.
set +u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate radfoam 2>/dev/null
cd ~/radfoam || exit 1
mkdir -p ~/rf20k

GPU=${1:-0}
SHARD=${2:-0}      # 0 = even index, 1 = odd index; two GPUs split the work
NSHARD=${3:-2}

SCENES="scene0062_00 scene0347_00 scene0070_00 scene0140_00 scene0645_00 scene0590_00 scene0000_00 scene0097_00 scene0200_00 scene0400_00"

i=0
for s in $SCENES; do
  for arm in match unfroz; do
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    tag="rf20k_${arm}_${s}"
    [ -f ~/rf20k/$tag.done ] && { echo "[skip] $tag"; continue; }
    [ -f output/$tag/model.pt ] && { touch ~/rf20k/$tag.done; echo "[have] $tag"; continue; }
    src=configs/scannet_$([ "$arm" = match ] && echo match || echo unfroz)_$s.yaml
    [ -f "$src" ] || src=configs/_tmpl_scannet_match.yaml
    cfg=configs/${tag}.yaml
    # Paper schedule. Everything else (init/final points, lr, sh, activation) inherited from the
    # 30k config so ONLY the schedule differs.
    sed -e "s/^iterations:.*/iterations: 20000/" \
        -e "s/^freeze_points:.*/freeze_points: 18000/" \
        -e "s/^densify_from:.*/densify_from: 2000/" \
        -e "s/^densify_until:.*/densify_until: 11000/" \
        -e "s/^experiment_name:.*/experiment_name: ${tag}/" \
        "$src" > "$cfg"
    grep -q "^experiment_name:" "$cfg" || echo "experiment_name: ${tag}" >> "$cfg"
    FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
    if [ "$FREE" -lt 12 ]; then echo "[STOP] only ${FREE}G free"; exit 1; fi
    echo "[train] gpu$GPU $tag free=${FREE}G $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU python train.py -c "$cfg" > ~/rf20k/$tag.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f output/$tag/model.pt ]; then
      touch ~/rf20k/$tag.done
      echo "  ok $(grep -oE 'Average PSNR: [0-9.]+' output/$tag/metrics.txt 2>/dev/null | tail -1)"
    else
      echo "  FAILED rc=$rc"; tail -3 ~/rf20k/$tag.log
    fi
  done
done
echo "[RF20K SHARD $SHARD DONE] $(date +%H:%M:%S)"
