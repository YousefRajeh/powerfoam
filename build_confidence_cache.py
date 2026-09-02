"""Per-primitive lifting confidences, aligned to the rows of the decision cache's `unit`."""
import os
import sys

sys.path.insert(0, r"D:\Downloads\feature-foam-lifting\src")
sys.path.insert(0, r"D:\Downloads\powerfoam")

import numpy as np
import torch

from probe_lifting_confidence import confidences
from run_cluster_classify_eval import SCENES

CACHE = r"C:\Users\rajehyl\AppData\Local\Temp\claude\D--Downloads\e7a70b1a-5e51-4a17-88b9-35aadc8d5988\scratchpad\dcache"

if __name__ == "__main__":
    suffix = "_ogl3"
    for scene in SCENES:
        out = os.path.join(CACHE, f"conf_{scene}{suffix}.pt")
        if os.path.exists(out):
            continue
        vm = torch.load(f"artifacts/scannet/{scene}/solved_geometric_median_nonfrozen{suffix}.pt",
                        map_location="cpu", weights_only=True)["valid_mask"].cpu().numpy()
        idx = torch.from_numpy(np.where(vm)[0])
        conf = {k: v.cpu()[idx] for k, v in confidences(scene, suffix, device="cpu").items()}
        torch.save(conf, out)
        print(scene, {k: float(v.mean()) for k, v in conf.items()}, flush=True)
