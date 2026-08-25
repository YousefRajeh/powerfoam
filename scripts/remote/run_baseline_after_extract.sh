#!/usr/bin/env bash
# Regenerate the 10-scene WHITE-FILL baseline through the SAME eval code that scores the
# black-fill run, so the final comparison isolates the fill colour rather than pipeline drift.
#
# Eval-only: all 10 solved_geometric_median_nonfrozen_l3.pt already exist, so no lift and no
# re-extraction. Waits for extraction to finish first -- this must not contend with it.
set +u
cd /d/Downloads/powerfoam
until grep -q "ALL DONE" logs_l3_runner.log 2>/dev/null; do sleep 120; done
echo "[baseline] extraction finished, starting white-fill 10-scene eval $(date +%H:%M:%S)"
FEAT_SUFFIX=_l3 /d/conda/envs/powerfoam/python.exe -u run_cluster_classify_eval.py \
  > logs_baseline_l3_10scene.log 2>&1
echo "[baseline] rc=$? $(date +%H:%M:%S)"
tail -6 logs_baseline_l3_10scene.log
