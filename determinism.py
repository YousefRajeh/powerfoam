"""Make the evaluation path bitwise reproducible.

WHY THIS EXISTS

Evaluating the IDENTICAL solved-feature file twice gave 36.12 and 36.60 mIoU at 19 classes on
scene0347_00. That ~0.5 mIoU spread is a noise floor large enough to swallow most of the effects
this project measures -- several ideas were scored, correctly, as "null" at deltas smaller than
that, and one single-scene "+1.72" needed to be discounted against it.

The cause is NOT the seeds, which were already set. It is float accumulation order. Measured
directly (2M rows, 320 labels, 512 channels):

    index_add_ default:              two identical calls differ, max |delta| = 4.3e-04
    index_add_ under deterministic:  two identical calls are BITWISE identical

`index_add_` on CUDA uses atomics, so the summation order varies run to run. k-means centroid
updates are built on it and run for 25 iterations, so a 1e-4 perturbation compounds until a
point near a centroid boundary flips cluster. That relabels a whole leaf, and the leaf's pooled
feature then classifies differently, which moves every GT point the leaf owns. A float-noise
difference becomes a whole-region label change -- which is exactly why the effect is ~0.5 mIoU
rather than ~0.001.

`torch.use_deterministic_algorithms(True)` selects a deterministic index_add_ implementation and
fixes it. cdist was verified already deterministic.

USAGE: call `enable_determinism()` at the top of any evaluation entry point, BEFORE any CUDA
work. `warn_only=True` so an unsupported op degrades to a warning rather than killing a long
run -- but that also means enabling this is not by itself proof of reproducibility. Verify by
running the same input twice and comparing (verify_determinism.py does this end to end).

CUBLAS_WORKSPACE_CONFIG must be set before the cuBLAS handle is created, i.e. before the first
matmul. Setting it here works when enable_determinism() is called early; setting it in the shell
before launching is strictly safer and is what the runner scripts do.
"""
import os
import random

_ENABLED = False


def enable_determinism(seed: int = 0, warn_only: bool = True, verbose: bool = True):
    """Seed everything and force deterministic kernels. Idempotent."""
    global _ENABLED
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    _ENABLED = True
    if verbose:
        print(f"[determinism] enabled (seed={seed}, warn_only={warn_only}, "
              f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']})", flush=True)


def is_enabled() -> bool:
    return _ENABLED
