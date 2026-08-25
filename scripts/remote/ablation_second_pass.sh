#!/usr/bin/env bash
# Second ablation pass: pick up the weighted solves and Gaussian distills that did not exist
# when the first pass walked each scene. The runner reads features at the moment it reaches a
# scene, so arms whose files appeared later are simply absent from pass one.
#
# This is cheap rather than a redo: DB.have_result skips every (scene, recon, features,
# solver, grouping, class_set) cell already recorded, and the assignments and adjacency graphs
# are cached, so only genuinely new cells cost anything.
set +u
cd /d/Downloads/powerfoam
until grep -q "results so far" logs_ablation_full.log 2>/dev/null; do sleep 120; done
echo "[pass2] first pass finished $(date +%H:%M:%S)"
/d/conda/envs/powerfoam/python.exe -u run_ablation.py \
  --note "pass 2: weighted solver + gaussian arms" > logs_ablation_pass2.log 2>&1
echo "[pass2] done rc=$? $(date +%H:%M:%S)"
/d/conda/envs/powerfoam/python.exe make_ablation_table.py
