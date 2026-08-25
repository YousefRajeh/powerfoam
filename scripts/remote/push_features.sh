#!/usr/bin/env bash
# Push locally-extracted L3 features to the remote, which has a working radfoam build.
# The radfoam CUDA extension cannot be imported locally: the only local build
# (D:\Downloads\foamvol) is the cryo-ET fork, exposing create_ct_pipeline instead of
# create_pipeline, so RadFoamScene cannot be constructed here. Rebuilding upstream radfoam on
# Windows needs CUDA 11.8 + MSVC 14.44 workarounds; the remote already has it working and
# holds every radfoam checkpoint natively.
K=/c/Users/rajehyl/.ssh/id_ed25519_powerfoam
H=rajehyl@10.68.106.173
cd /d/Downloads/powerfoam/data/scannet
for s in scene0062_00 scene0347_00 scene0070_00 scene0097_00 scene0200_00 scene0140_00 scene0645_00; do
  d=${s}_colmap/openclip_features_sam_l3
  [ -d "$d" ] || continue
  n=$(ls "$d" | wc -l)
  r=$(ssh -i $K -o ConnectTimeout=25 $H "ls ~/powerfoam/data/scannet/$d 2>/dev/null | wc -l")
  if [ "$r" -ge "$n" ]; then echo "[skip] $s ($r/$n already there)"; continue; fi
  echo "[push] $s ($n files, $(du -sm "$d" | cut -f1) MB) $(date +%H:%M:%S)"
  ssh -i $K $H "mkdir -p ~/powerfoam/data/scannet/$d"
  scp -q -i $K "$d"/* "$H:~/powerfoam/data/scannet/$d/" && echo "  ok" || echo "  FAILED"
done
echo "[PUSH DONE] $(date +%H:%M:%S)"
