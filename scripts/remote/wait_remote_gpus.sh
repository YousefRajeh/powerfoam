#!/usr/bin/env bash
# Poll the remote until NO compute processes owned by other users remain, then exit.
# Colleagues engeld (GPU0) and lid0e (GPU1/GPU2) hold the GPUs; the user chose to wait for
# them rather than contend. Counts only foreign PIDs, so our own jobs never self-trigger.
K=/c/Users/rajehyl/.ssh/id_ed25519_powerfoam
H=rajehyl@10.68.106.173
for i in $(seq 1 96); do          # 96 * 5 min = 8 h ceiling
  n=$(ssh -i $K -o ConnectTimeout=25 $H 'c=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do u=$(ps -o user= -p $p 2>/dev/null); [ -n "$u" ] && [ "$u" != "rajehyl" ] && c=$((c+1)); done; echo $c' 2>/dev/null)
  [ -z "$n" ] && n=99            # ssh hiccup: treat as still busy, never as free
  if [ "$n" = "0" ]; then
    echo "REMOTE GPUS FREE after ~$((i*5)) min"
    ssh -i $K -o ConnectTimeout=25 $H 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
    exit 0
  fi
  sleep 300
done
echo "gave up waiting after 8h; foreign procs still present"
