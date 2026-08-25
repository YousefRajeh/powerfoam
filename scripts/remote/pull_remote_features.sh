#!/usr/bin/env bash
# Bring back the two scenes the remote GPUs extracted (scene0000_00, scene0590_00) so the
# local scoring pipeline and ablation see all ten. Waits for each to be complete remotely.
K=/c/Users/rajehyl/.ssh/id_ed25519_powerfoam
H=rajehyl@10.68.106.173
cd /d/Downloads/powerfoam/data/scannet
for s in scene0590_00 scene0000_00; do
  d=${s}_colmap/openclip_features_sam_l3
  need=$(( 2 * $(ls ${s}_colmap/images | wc -l) ))
  for i in $(seq 1 120); do
    r=$(ssh -i $K -o ConnectTimeout=25 $H "ls ~/powerfoam/data/scannet/$d 2>/dev/null | wc -l")
    [ "$r" -ge "$need" ] && break
    echo "[wait] $s remote $r/$need"; sleep 120
  done
  mkdir -p "$d"
  have=$(ls "$d" 2>/dev/null | wc -l)
  [ "$have" -ge "$need" ] && { echo "[skip] $s already local"; continue; }
  echo "[pull] $s $(date +%H:%M:%S)"
  scp -q -i $K "$H:~/powerfoam/data/scannet/$d/*" "$d/" && echo "  ok ($(ls $d | wc -l)/$need)" || echo "  FAILED"
done
echo "[PULL DONE] $(date +%H:%M:%S)"
