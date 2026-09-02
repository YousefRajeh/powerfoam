#!/usr/bin/env bash
# Matched held-out PSNR: radfoam vs powerfoam, same scenes, same split.
#
# WHY A RETRAIN IS UNAVOIDABLE. Every existing checkpoint on both sides was trained with the
# holdout DISABLED (powerfoam `eval: false` -> train_split="all"; radfoam `no_holdout: true`),
# so no view is unseen and no held-out number can be extracted from them. The train-set PSNRs
# already measured (pf_tfroz 27.71 vs rf_unfroz 33.01, +6.17 dB to radfoam over 8 scenes) are
# apples-to-apples -- both sides used the same protocol -- but both are inflated, and inflated
# MOST for the highest-capacity model, since rf_unfroz fits 3x the points to the very views it
# is scored on. This run tests whether that 6 dB survives on unseen views.
#
# The split rule is identical in both codebases (indices % 8 == 0 -> test), so enabling the
# holdout on each side yields the same partition without any new code.
#
# SCOPE: 3 scenes x 4 arms. Not all ten -- this answers "is the gap real or overfitting",
# which does not need ten scenes, and the GPUs are shared with two other users.
set +u
K=/c/Users/rajehyl/.ssh/id_ed25519_powerfoam
H=rajehyl@10.68.106.173
SCENES="scene0062_00 scene0347_00 scene0070_00"
ssh -i $K -o ConnectTimeout=30 $H "bash -s" <<'REMOTE'
set +u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate radfoam 2>/dev/null
cd ~/radfoam
mkdir -p ~/heldout
for s in scene0062_00 scene0347_00 scene0070_00; do
  for arm in match unfroz; do
    tag="ho_rf_${arm}_${s}"
    [ -f ~/heldout/$tag.done ] && { echo "[skip] $tag"; continue; }
    src=configs/scannet_$([ "$arm" = match ] && echo match || echo unfroz)_$s.yaml
    [ -f "$src" ] || { echo "[miss] $tag: no config $src"; continue; }
    cfg=configs/${tag}.yaml
    # flip the holdout on; everything else identical to the run that produced model.pt
    sed -e "s/^no_holdout:.*/no_holdout: false/" -e "s/^experiment_name:.*/experiment_name: ${tag}/" "$src" > "$cfg"
    grep -q "^no_holdout:" "$cfg" || echo "no_holdout: false" >> "$cfg"
    echo "[train] $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=1 python train.py -c "$cfg" > ~/heldout/$tag.log 2>&1 \
      && touch ~/heldout/$tag.done && echo "  ok $(grep -oE 'Average PSNR: [0-9.]+' output/$tag/metrics.txt 2>/dev/null | tail -1)" \
      || echo "  FAILED rc=$?"
  done
done
echo "[RADFOAM HELDOUT DONE]"
REMOTE
