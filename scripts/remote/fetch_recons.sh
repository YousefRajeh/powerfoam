#!/usr/bin/env bash
# Pull all remote reconstructions locally for the ablation. ~6.7 GB total.
# Layout mirrors what ablation_assign.py expects:
#   recon_remote/{rf_froz,rf_unfroz,gs_froz,gs_unfroz}/<scene>/<file>
K=/c/Users/rajehyl/.ssh/id_ed25519_powerfoam
H=rajehyl@10.68.106.173
cd /d/Downloads/powerfoam/recon_remote
for s in scene0062_00 scene0347_00 scene0070_00 scene0140_00 scene0645_00 \
         scene0590_00 scene0000_00 scene0097_00 scene0200_00 scene0400_00; do
  mkdir -p rf_froz/$s rf_unfroz/$s gs_froz/$s gs_unfroz/$s
  for spec in "rf_froz/$s/model.pt:radfoam/output/rf_match_$s/model.pt" \
              "rf_unfroz/$s/model.pt:radfoam/output/rf_unfroz_$s/model.pt" \
              "gs_froz/$s/ckpt.pt:gaussian_baseline_scannet/$s/ckpts/ckpt_29999_rank0.pt" \
              "gs_unfroz/$s/ckpt.pt:gaussian_unfrozen_scannet/$s/ckpts/ckpt_29999_rank0.pt"; do
    dst=${spec%%:*}; src=${spec#*:}
    [ -s "$dst" ] && continue
    scp -q -i $K "$H:~/$src" "$dst" 2>/dev/null && echo "[ok] $dst ($(du -m "$dst" | cut -f1) MB)" || echo "[MISS] $dst"
  done
done
echo "[FETCH DONE] $(du -sh /d/Downloads/powerfoam/recon_remote | cut -f1)"
