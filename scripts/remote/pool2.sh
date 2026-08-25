#!/bin/bash
# Shared work-stealing pool for the 6-arm segmentation ablation. v2: adds the pftfroz arm and
# fixes two self-inflicted bugs from v1.
#
# BUG 1 (v1): `set -u`. conda's activate.d/activate-gcc_linux-64.sh references an unbound
#   SYS_SYSROOT, so every worker died instantly at `conda activate`. Never use nounset here.
# BUG 2 (v1): the startup loop waited for ANY `train.py` process to exit before pulling work.
#   That deadlocks any worker added while its siblings are busy. Pool claiming + per-worker
#   CUDA_VISIBLE_DEVICES already prevent collisions, so no wait is needed at all.
#
# Claiming is atomic via mkdir (single POSIX filesystem op) -- two workers cannot take the same
# job. Jobs are ordered longest-first (LPT), so the ~10-minute gaussian runs act as tail fillers
# and the GPUs land within one short job of each other rather than hours apart.
#
# usage: pool2.sh <gpu>
set +u
GPU=$1
Q=~/ablation_queue
source ~/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=$GPU

declare -A N=( [scene0000_00]=81369 [scene0062_00]=51610 [scene0070_00]=109380 \
               [scene0097_00]=72007 [scene0140_00]=372941 [scene0200_00]=83291 \
               [scene0347_00]=67984 [scene0400_00]=155959 [scene0590_00]=222957 \
               [scene0645_00]=352477 )

echo "[gpu$GPU] pool2 worker up $(date -Is)"

while :; do
  CLAIMED=""
  for j in $(ls "$Q/jobs" 2>/dev/null | sort); do
    if mkdir "$Q/claimed/$j" 2>/dev/null; then CLAIMED="$j"; break; fi
  done
  if [ -z "$CLAIMED" ]; then echo "[gpu$GPU] pool empty $(date -Is)"; break; fi

  ARM=$(echo "$CLAIMED" | cut -d_ -f2)
  SC=$(echo "$CLAIMED" | cut -d_ -f3-)
  NPTS=${N[$SC]}
  FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 20 ]; then
    echo "[gpu$GPU][STOP] only ${FREE}G free"; rmdir "$Q/claimed/$CLAIMED"; break
  fi
  echo "[gpu$GPU][START] $ARM $SC free=${FREE}G $(date -Is)"
  T0=$(date +%s); RC=99

  case "$ARM" in
    rffroz|rfunfroz)
      conda activate radfoam
      export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
      cd ~/radfoam || exit 1
      if [ "$ARM" = "rffroz" ]; then
        TAG=rf_match_$SC; CFG=configs/scannet_match_$SC.yaml
        sed -e "s/scene0062_00_colmap/${SC}_colmap/" \
            -e "s/^init_points:.*/init_points: ${NPTS}/" \
            -e "s/^final_points:.*/final_points: ${NPTS}/" \
            -e "s/^experiment_name:.*/experiment_name: ${TAG}/" \
            configs/_tmpl_scannet_match.yaml > "$CFG"
      else
        TAG=rf_unfroz_$SC; CFG=configs/scannet_unfroz_$SC.yaml
        FINAL=$(( NPTS * 3 ))
        # freeze_points=28000 follows arXiv:2502.01157 Sec.4: "the last 2k only updating
        # radiance and density attributes while positions are frozen" (their 20k -> 18000).
        sed -e "s/scene0062_00_colmap/${SC}_colmap/" \
            -e "s/^init_points:.*/init_points: ${NPTS}/" \
            -e "s/^final_points:.*/final_points: ${FINAL}/" \
            -e "s/^experiment_name:.*/experiment_name: ${TAG}/" \
            -e "s/^frozen_init_pts:.*/frozen_init_pts: false/" \
            -e "s/^points_lr_init:.*/points_lr_init: 2.0e-4/" \
            -e "s/^points_lr_final:.*/points_lr_final: 5.0e-6/" \
            -e "s/^freeze_points:.*/freeze_points: 28000/" \
            configs/_tmpl_scannet_match.yaml > "$CFG"
      fi
      if [ ! -s "$CFG" ]; then
        echo "[gpu$GPU][ERR] generated $CFG is empty"; RC=97
      elif [ -f "output/$TAG/model.pt" ]; then
        echo "[gpu$GPU][SKIP] $TAG"; RC=0
      else
        python train.py -c "$CFG" > ~/train_${ARM}_${SC}.log 2>&1; RC=$?
        rm -rf "output/$TAG/test" 2>/dev/null
      fi
      ;;

    pftfroz)
      conda activate powerfoam
      cd ~/powerfoam || exit 1
      OUT=output/scannet_${SC}_truefrozen
      if [ -f "$OUT/model.pt" ]; then
        echo "[gpu$GPU][SKIP] $SC"; RC=0
      else
        CFG=configs/_tf_${SC}.yaml
        sed -e "s/^scene:.*/scene: ${SC}_colmap/" \
            -e "s/^init_points:.*/init_points: ${NPTS}/" \
            -e "s/^final_points:.*/final_points: ${NPTS}/" \
            configs/scannet_truefrozen.yaml > "$CFG"
        python train.py -c "$CFG" --experiment_name "scannet_${SC}_truefrozen" \
          > ~/train_pftfroz_${SC}.log 2>&1; RC=$?
      fi
      ;;

    gsfroz)
      source ~/gs_train_venv/bin/activate
      export CUDA_HOME=/home/rajehyl/miniconda3/envs/powerfoam
      export PATH=$CUDA_HOME/bin:$PATH
      OUT=~/gaussian_baseline_scannet/$SC
      if [ -f "$OUT/ckpts/ckpt_29999_rank0.pt" ]; then
        echo "[gpu$GPU][SKIP] $SC"; RC=0
      else
        # Frozen: positions fixed (means_lr 0) and densification off (refine-stop-iter 0),
        # matching the existing gaussian_baseline_scannet/ cfg.yml for the other 9 scenes.
        python ~/splat-distiller/gaussian_splatting/simple_trainer.py default \n          --data_dir /home/rajehyl/powerfoam/data/scannet/${SC}_colmap \n          --data_factor 1 --result_dir "$OUT" \n          --init_type sfm --max_steps 30000 \n          --means_lr 0.0 --strategy.refine-stop-iter 0 \n          --eval_steps 1000000000 --ply_steps 30000 \n          --disable_viewer --disable_video > ~/train_gsfroz_${SC}.log 2>&1; RC=$?
      fi
      ;;
    gsunfroz)
      source ~/gs_train_venv/bin/activate
      # nvcc ships at <env>/bin/nvcc and cicc at <env>/nvvm/bin/cicc; a CUDA_HOME of
      # <env>/targets/x86_64-linux makes nvcc look for cicc one level too deep and every
      # JIT build dies with 'cicc: command not found' (rc=127).
      export CUDA_HOME=/home/rajehyl/miniconda3/envs/powerfoam
      export PATH=$CUDA_HOME/bin:$PATH
      OUT=~/gaussian_unfrozen_scannet/$SC
      if [ -f "$OUT/ckpts/ckpt_29999_rank0.pt" ]; then
        echo "[gpu$GPU][SKIP] $SC"; RC=0
      else
        # Identical to the frozen gaussian arm except the two freeze knobs are released:
        # means_lr 0.0 -> 1.6e-4, refine_stop_iter 0 -> 15000.
        python ~/splat-distiller/gaussian_splatting/simple_trainer.py default \
          --data_dir /home/rajehyl/powerfoam/data/scannet/${SC}_colmap \
          --data_factor 1 --result_dir "$OUT" \
          --init_type sfm --max_steps 30000 \
          --means_lr 1.6e-4 --strategy.refine-stop-iter 15000 \
          --eval_steps 1000000000 --ply_steps 30000 \
          --disable_viewer --disable_video > ~/train_gsunfroz_${SC}.log 2>&1; RC=$?
      fi
      ;;
    *)
      echo "[gpu$GPU][ERR] unknown arm '$ARM'"; RC=98
      ;;
  esac

  DT=$(( $(date +%s) - T0 ))
  echo "[gpu$GPU][DONE ] $ARM $SC rc=$RC $((DT/60))min $(date -Is)"
  if [ "$RC" -eq 0 ]; then
    touch "$Q/done/$CLAIMED"; rm -f "$Q/jobs/$CLAIMED"
  else
    # Failed jobs go back in the pool rather than being silently marked complete.
    N=$(cat "$Q/attempts_$CLAIMED" 2>/dev/null || echo 0); N=$((N+1))
    echo "$N" > "$Q/attempts_$CLAIMED"
    if [ "$N" -ge 3 ]; then
      echo "[gpu$GPU][GIVEUP] $CLAIMED after $N attempts rc=$RC"
      mkdir -p "$Q/failed"; mv "$Q/jobs/$CLAIMED" "$Q/failed/$CLAIMED" 2>/dev/null
      rmdir "$Q/claimed/$CLAIMED" 2>/dev/null
    else
      echo "[gpu$GPU][REQUEUE] $CLAIMED rc=$RC attempt=$N"; rmdir "$Q/claimed/$CLAIMED" 2>/dev/null
    fi
    touch "$Q/failed_${CLAIMED}"
  fi
done
